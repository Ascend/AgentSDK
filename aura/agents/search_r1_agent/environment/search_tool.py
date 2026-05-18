#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os
import time
import uuid
from typing import Any, Optional
import requests
from rllm.tools.tool_base import Tool, ToolOutput

from aura.base.log.loggers import Loggers

logger = Loggers(__name__).get_logger()


DEFAULT_TIMEOUT = 30  # Default search request timeout
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1
API_TIMEOUT = 10


def call_search_api(
    retrieval_service_url: str,
    query_list: list[str],
    topk: int = 3,
    return_scores: bool = True,
    timeout: int = DEFAULT_TIMEOUT,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    """
    Calls the remote search API to perform retrieval with retry logic for various errors,
    using increasing delay between retries. Logs internal calls with a unique ID.

    Args:
        retrieval_service_url: The URL of the retrieval service API.
        query_list: List of search queries.
        topk: Number of top results to return.
        return_scores: Whether to return scores.
        timeout: Request timeout in seconds.

    Returns:
        A tuple (response_json, error_message).
        If successful, response_json is the API's returned JSON object, error_message is None.
        If failed after retries, response_json is None, error_message contains the error information.
    """
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = {"queries": query_list, "topk": topk, "return_scores": return_scores}

    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
            )
            response = requests.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt "
                    f"{attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            logger.info(f"{log_prefix}Search API call successful on attempt {attempt + 1}")
            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If loop finishes without returning success, return the last recorded error
    logger.error(f"{log_prefix}Search API call failed. Last error: {last_error}")
    return None, last_error.replace(log_prefix, "API Call Failed: ") if last_error else "API Call Failed after retries"


def _passages2string(retrieval_result):
    """Convert retrieval results to formatted string."""
    format_reference = ""
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item["document"]["contents"]
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        format_reference += f"Doc {idx + 1} (Title: {title})\n{text}\n\n"
    return format_reference.strip()


class SearchTool(Tool):
    """
    display
    """

    def __init__(self, name: str = "search", description: str = "Searches for relevant information based on queries."):
        """
        Initialize the Local Retrieval Tool.

        Args:
            name: Tool name
            description: Tool description
        """
        super().__init__(name=name, description=description)
        self.retrieval_service_url = os.getenv("SEARCH_R1_SERVICE_URL", "http://0.0.0.0:8000/retrieve")
        self.topk = os.getenv("SEARCH_R1_TOPK", 3)
        self.return_scores = os.getenv("SEARCH_R1_RETURN_SCORES", True)
        self.timeout = os.getenv("SEARCH_R1_TIMTOUT", 30)

    @property
    def json(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "arguments": {
                    "type": "object",
                    "properties": {
                        "query_list": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of search queries",
                        }
                    },
                    "required": ["query_list"],
                },
            },
        }

    def forward(self, query_list: list[str], **kwargs) -> ToolOutput:
        """Execute the search tool.

        Args:
            parameters: Tool parameters containing query_list and optional timeout

        Returns: tool_response, tool_reward_score, tool_metrics
            tool_response: The response str of the tool.
            tool_reward_score: The step reward score of the tool.
            tool_metrics: The metrics of the tool.
        """
        logger.info(f"SearchTool forward parameters: {query_list}")
        metadata = dict()
        query_list_from_params = query_list
        if not query_list_from_params or not isinstance(query_list_from_params, list):
            error_msg = "Error: 'query_list' is missing, empty, or not a list in parameters."
            logger.error(f"[SearchTool] {error_msg} Received parameters: {query_list}")
            return ToolOutput(name=self.name, output=json.dumps({"result": error_msg}))
        try:
            start_time = time.perf_counter()
            logger.info(f"SearchTool forward retrieval_service_url: {self.retrieval_service_url}")
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.retrieval_service_url,
                query_list=query_list_from_params,
                topk=self.topk,
                return_scores=self.return_scores,
                timeout=self.timeout,
            )
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f"elapsed_time {elapsed_time}, response: {api_response}")
            logger.info(f"api_response type: {type(api_response)}")
            if error_msg:
                logger.error(f"[SearchTool] Search API call failed: {error_msg}")
                return ToolOutput(name=self.name, output=json.dumps({"result": error_msg}), metadata=metadata)
            if api_response and isinstance(api_response, dict):
                raw_results = api_response.get("result", [])
                logger.info(f"raw_results: {raw_results}")
                if raw_results:
                    pretty_results = []
                    for retrieval in raw_results:
                        logger.info(f"retrieval: {retrieval}")
                        formatted = _passages2string(retrieval)
                        logger.info(f"formatted: {formatted}")
                        pretty_results.append(formatted)
                    final_result = "\n---\n".join(pretty_results)
                    logger.info(f"SearchTool forward final_result: {final_result}")
                    return ToolOutput(name=self.name, output=final_result, metadata=metadata)
            return ToolOutput(name=self.name, output="", metadata=metadata)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Unexpected error: {str(e)}")


def get_search_r1_tools():
    tool_map = {"search": SearchTool}
    return tool_map
