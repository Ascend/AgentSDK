#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2025 Alibaba-NLP.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import asyncio
import os
import re
import urllib.parse
from typing import Any

from agents.webwalker_agent.cache import get_page_cache_store, normalize_cache_url
from agents.webwalker_agent.cache.page_cache_store import (
    CACHE_MODE_OFF,
    CACHE_MODE_READ_ONLY,
    CACHE_MODE_READ_WRITE,
    CACHE_MODE_STRICT,
    VALID_CACHE_MODES,
)
from agents.webwalker_agent.environment.runtime import crawl_lock, safe_asyncio_run
import logging

logger = logging.getLogger(__name__)


class WebWalkerPageFetchMixin:
    # ========================== Page parsing ==========================
    def process_url(self, base_url, sub_url):
        return urllib.parse.urljoin(base_url, sub_url)

    def clean_markdown(self, res):
        pattern = r'\[.*?\]\(.*?\)'
        try:
            result = re.sub(pattern, '', res)
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            result = re.sub(url_pattern, '', result)
            result = result.replace("* \n", "")
            result = re.sub(r"\n\n+", "\n", result)
            return result
        except Exception:
            return res

    @staticmethod
    def _resolve_cache_mode(raw_mode: Any) -> str:
        """Resolve the persistent cache behaviour from kwargs/env, defaulting safely."""
        mode = str(
            raw_mode
            if raw_mode is not None
            else os.getenv("WEBWALKER_CACHE_MODE", "")
        ).strip().lower()
        if not mode:
            mode = CACHE_MODE_OFF
        if mode not in VALID_CACHE_MODES:
            logger.warning(
                f"[PAGE CACHE] Unknown cache_mode='{mode}', fallback to '{CACHE_MODE_OFF}'. "
                f"Valid modes: {VALID_CACHE_MODES}"
            )
            mode = CACHE_MODE_OFF
        return mode

    def _init_page_store(self):
        """Open the persistent page cache store according to the configured mode."""
        if self.cache_mode == CACHE_MODE_OFF or not self.page_cache_path:
            if self.cache_mode != CACHE_MODE_OFF and not self.page_cache_path:
                logger.warning(
                    f"[PAGE CACHE] cache_mode={self.cache_mode} but page_cache_path is missing; "
                    "persistent cache is disabled and live fetch will be used."
                )
            return None
        # strict/read_only open the store read-only; read_write can write back.
        read_only = self.cache_mode in (CACHE_MODE_STRICT, CACHE_MODE_READ_ONLY)
        store = get_page_cache_store(self.page_cache_path, read_only=read_only)
        if store is None:
            logger.warning(
                f"[PAGE CACHE] Failed to open cache store {self.page_cache_path}; live fetch will be used."
            )
        return store

    def _fetch_page_success_only(self, url, screenshot=False):
        cache_key = (url, bool(screenshot))
        cached = self._page_cache.get(cache_key)
        if cached is not None:
            return cached

        # Persistent cache reads apply to non-screenshot requests and avoid network access on hit.
        if self._page_store is not None and not screenshot:
            persisted = self._page_store.get(normalize_cache_url(url))
            if persisted is not None:
                self._page_cache[cache_key] = persisted
                return persisted
            if self.cache_mode == CACHE_MODE_STRICT:
                raise Exception(
                    f"[PAGE CACHE strict] cache miss for url (offline mode, no live fetch): {url}"
                )

        with crawl_lock:
            # Other beam branches may have fetched the same URL while this branch waited.
            cached = self._page_cache.get(cache_key)
            if cached is not None:
                return cached

            ret = safe_asyncio_run(self.get_info(url, screenshot=screenshot))
            if isinstance(ret, list) and len(ret) == 1:
                ret = ret[0]

            html, markdown = ret
            self._page_cache[cache_key] = (html, markdown)
            # Write back in read_write mode to warm the persistent cache.
            if (
                self._page_store is not None
                and not screenshot
                and self.cache_mode == CACHE_MODE_READ_WRITE
            ):
                try:
                    self._page_store.put(
                        normalize_cache_url(url),
                        html,
                        markdown,
                        status="ok",
                        root_url=self.root_url,
                    )
                except Exception as put_error:  # noqa: BLE001 - never break rollout on cache write
                    logger.warning(f"[PAGE CACHE] Write-back failed for {url}: {put_error}")
            return html, markdown

    async def get_info(self, url, screenshot=False, timeout=20.0):
        """Fetch page information with timeout handling.

        Args:
            url: Page URL to visit.
            screenshot: Whether to capture a screenshot.
            timeout: Timeout in seconds.

        Returns:
            tuple: HTML content and cleaned markdown content.
        """
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

            run_config = CrawlerRunConfig(screenshot=screenshot)
            async with AsyncWebCrawler() as crawler:
                result = await asyncio.wait_for(
                    crawler.arun(url, config=run_config),
                    timeout=timeout
                )
                return result.html, self.clean_markdown(result.markdown)
        except asyncio.TimeoutError:
            raise Exception(f"Request timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Failed to fetch page: {str(e)}")

    def extract_links_with_text(self, html):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, 'html.parser')
        links = []

        # Extract supported href and onclick link forms.
        for a_tag in soup.find_all('a', href=True):
            url = a_tag['href']
            text = ''.join(a_tag.stripped_strings)
            if text and "javascript" not in url and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
                full_url = self.process_url(self.root_url, url)
                if full_url.startswith(self.root_url):
                    links.append({'url': full_url, 'text': text})

        for a_tag in soup.find_all('a', onclick=True):
            onclick_text = a_tag['onclick']
            text = ''.join(a_tag.stripped_strings)
            match = re.search(r"window\.location\.href='([^']*)'", onclick_text)
            if match:
                url = match.group(1)
                if url and text and not url.endswith(('.jpg', '.png', '.gif', '.jpeg', '.pdf')):
                    full_url = self.process_url(self.root_url, url)
                    if full_url.startswith(self.root_url):
                        links.append({'url': full_url, 'text': text})

        # Deduplicate links and refresh the environment URL map.
        unique_links = {f"{item['url']}_{item['text']}": item for item in links}

        info = ""
        for i in list(unique_links.values()):
            self.button_url_dict[i["text"]] = i["url"]
            info += "<button>" + i["text"] + "</button>\n"

        return info
