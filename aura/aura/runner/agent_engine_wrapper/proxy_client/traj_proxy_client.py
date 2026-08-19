#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import random
from typing import List, Dict, Any
from datetime import datetime
from email.utils import parsedate_to_datetime
import asyncio
import aiohttp
import httpx

from aura.controllers.utils.http_status import HTTP_OK_200
from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()

MICROSECONDS_PER_SECOND = 1_000_000.0  # 微秒转秒的换算系数
SECONDS_PER_DAY = 24 * 60 * 60.0  # 每天秒数


def parse_datetime(time_str: str) -> datetime | None:
    """
    Parses a time string into a datetime object. Multiple formats are supported

    Args:
        time_str: time string, which can be in ISO or RFC 2822 format

    Returns:
        datetime | None: parsing datetime object. If the parsing fails, None is returned
    """
    if not time_str:
        return None

    try:
        return parsedate_to_datetime(time_str)
    except (ValueError, TypeError, AttributeError):
        pass

    try:
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None

def calculate_time_diff_seconds(start_time_str, end_time_str) -> float | None:
    """
    Calculates the difference (in seconds) between two time strings with microsecond precision

    Args:
        start_time_str: start time string
        end_time_str: end time string

    Returns:
        float | None: time difference (in seconds). If the calculation fails, 0 is returned
    """
    if not start_time_str or not end_time_str:
        return 0

    try:
        if isinstance(start_time_str, str):
            start_time = parse_datetime(start_time_str)
            end_time = parse_datetime(end_time_str)
        else:
            start_time = start_time_str
            end_time = end_time_str

        if start_time is None or end_time is None:
            return 0

        time_diff = end_time - start_time
        return time_diff.microseconds / MICROSECONDS_PER_SECOND + time_diff.days * SECONDS_PER_DAY + time_diff.seconds
    except Exception:
        return 0

class TrajProxyClient:
    def __init__(self, model_name: str, infer_url: str | list):
        self.model_name = model_name
        self.infer_url = infer_url
        timeout = httpx.Timeout(300.0, connect=60.0, read=300.0)
        self.client = httpx.AsyncClient(timeout=timeout, limits=httpx.Limits(max_connections=None, max_keepalive_connections=None))

    async def get_agent_trajectory(self, session_id: str) -> dict[str, Any] | None:
        """
        Fetch trajectory records via HTTP request.

        Args:
            session_id: Session ID, corresponding to the session_id in the HTTP interface.

        Returns:
            dict: A dictionary containing session_id and a list of step_info. Returns None on failure.
        """
        if isinstance(self.infer_url, list):
            base_url = random.choice(self.infer_url)
        else:
            base_url = self.infer_url

        url = f"{base_url}/trajectory?session_id={session_id}&fields=-messages"
        logger.info(f"Getting trajectory, url: {url}, session_id: {session_id}")

        max_retries = 3
        headers = {
            "Accept-Encoding": "gzip, deflate"
        }
        for attempt in range(max_retries):
            try:
                response = await self.client.get(url, headers=headers)
                if response.status_code == HTTP_OK_200:
                    data = response.json()
                    records = data.get("records", [])
                    count = data.get("count", len(records))
                    logger.info(f"Get trajectory success, session_id: {session_id}, count: {count}")

                    step_infos = self._convert_step_info(records, session_id=session_id)
                    if len(step_infos) == 0:
                        logger.warning(f"Get trajectory, step_infos is empty, session_id: {session_id}")
                        return None

                    return {
                        "session_id": session_id,
                        "step_info": step_infos
                    }
                else:
                    logger.warning(f"Failed to get trajectory, session_id: {session_id}, HTTP {response.status_code}.")
            except Exception as e:
                logger.exception(f"Error getting trajectory, session_id: {session_id}, error: {repr(e)}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + 3
                logger.warning(f"Failed to get trajectory, session_id: {session_id}, retry {attempt + 1}/{max_retries} after {wait_time}s")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Failed to get trajectory after {max_retries} retries, session_id: {session_id}")
        return None

    def _get_system_prompt(self, messages: List[Dict[str, Any]]) -> str | None:
        for message in messages:
            if message.get("role") == "system":
                return message.get("content")
        return None

    def _get_model_response(self, messages: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        """
        Get the LLM output.
        Args:
            messages: List of message dictionaries.

        Returns:
            Dict[str, Any] | None: The assistant message dictionary.
        """
        for message in reversed(messages):
            role = message.get("role", None)
            if role == "assistant":
                return message
        return None

    def _get_env_response(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get the tool outputs.
        Args:
            messages: List of message dictionaries.

        Returns:
            List[Dict[str, Any]]: List of tool messages.
        """
        env_messages = []
        for message in reversed(messages):
            role = message.get("role", None)
            if role in ["assistant", "system"]:
                break
            elif role == "tool":
                env_messages.insert(0, message)
        return env_messages

    def _filter_invalid_records(self, records, session_id):
        # 1. Filter out records where response_text is "\n\nSAFE"
        filtered_safe_records = [record for record in records if record['response_text'] != "\n\nSAFE"]
        remove_safe = len(records) - len(filtered_safe_records)

        filtered_records = []
        for record in filtered_safe_records:
            raw = record.get('raw_response')
            # 2. Filter out records where raw_response is empty
            if raw is None:
                logger.info(f"{session_id=} filter record.raw_response is None")
                continue
            filtered_records.append(record)

        logger.info(f"{session_id=} filtered special/no raw_response records count: {len(filtered_records)}, remove rows:"
            f" {len(filtered_safe_records) - len(filtered_records)} remove_safe:{remove_safe}")
        return filtered_records

    def _convert_step_info(self, records: List[Dict[str, Any]], session_id: str) -> List[Dict[str, Any]]:
        """
        Convert trajectory records into step information.

        Args:
            records: List of records returned from TrajProxy.

        Returns:
            List[Dict[str, Any]]: Converted list of step information, including fields like step_id, env_time, llm_time, etc.
        """
        if not records:
            return []

        filtered_records = self._filter_invalid_records(records, session_id)
        # Sort by start_time field in ascending order
        sorted_records = sorted(filtered_records, key=lambda x: x.get('start_time', 0))

        # Add new attributes for each row
        result = []
        for i, record in enumerate(sorted_records):
            # Create a copy of the record to avoid modifying the original data
            step_info = record.copy()

            # Add step_id, starting from 0
            step_info['step_id'] = i

            # Add llm_time: end_time of current row - start_time of current row
            start_time = record.get('start_time')
            end_time = record.get('end_time')
            step_info['llm_time'] = calculate_time_diff_seconds(start_time, end_time)

            raw_request = step_info['raw_request']

            # Add system_prompt field, value is the content where role is system in messages
            system_prompt = self._get_system_prompt(raw_request.get('messages'))
            if system_prompt is not None:
                step_info['system_prompt'] = system_prompt
                truncated_prompt = system_prompt if len(system_prompt) <= 50 else system_prompt[:50] + "......"
                logger.debug(f"Get system_prompt: session_id={session_id}, step_id={i}: {truncated_prompt}")

            # Add model_response field, filled with raw_response of the current round
            step_info['model_response'] = step_info['raw_response']['choices'][0]['message']
            tools = step_info['raw_request'].get('tools', [])
            step_info['tools'] = tools
            logger.debug(f"Get tools: session_id={session_id}, step_id={i}, tools count={len(tools)}: {[tool['function']['name'] for tool in tools]}")

            if i > 0:
                # Add env_time: start_time of current row - end_time of previous row
                prev_end_time = sorted_records[i - 1].get('end_time')
                curr_start_time = record.get('start_time')
                result[i - 1]['env_time'] = calculate_time_diff_seconds(prev_end_time, curr_start_time)

                # Add env_response
                env_responses = self._get_env_response(raw_request.get('messages'))
                if env_responses is not None:
                    logger.debug(f"tool call counts: session_id={session_id}, call numbers={len(env_responses)}")
                    result[i - 1]['env_response'] = env_responses

            result.append(step_info)

        return result

    async def get_records_by_session(self, session_id):
        url = f"{self.infer_url}/trajectory?session_id={session_id}"
        logger.info(f"Getting trajectory, url: {url}, session_id: {session_id}")

        max_retries = 3
        headers = {
            "Accept-Encoding": "gzip, deflate"
        }
        for attempt in range(max_retries):
            try:
                response = await self.client.get(url, headers=headers)
                if response.status_code == HTTP_OK_200:
                    data = response.json()
                    records = data.get("records", [])
                    count = data.get("count", len(records))
                    logger.info(f"Get trajectory success, session_id: {session_id}, count: {count}")
                    return records
                else:
                    if attempt < max_retries - 1:
                        wait_time = attempt + 1
                        logger.warning(
                            f"Failed to get trajectory, session_id: {session_id}, HTTP {response.status_code}. "
                            f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(
                            f"Failed to get trajectory after {max_retries} retries, session_id: {session_id}, HTTP {response.status_code}")
                        return []
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = attempt + 1
                    logger.warning(f"Error getting trajectory, url: {url}, session_id: {session_id}, error: {e}. "
                                   f"Retry {attempt + 1}/{max_retries} after {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"Error getting trajectory after {max_retries} retries, url: {url}, session_id: {session_id}, error: {e}",
                        exc_info=True)
                    return []
        return []
