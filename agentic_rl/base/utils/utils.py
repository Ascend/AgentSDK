#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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
import datetime
import os
import socket
import subprocess
import time
from collections import defaultdict
from typing import List, re

import openai
import torch
import torch.distributed as dist
import vertexai
from google.cloud.aiplatform_v1beta1.types.content import SafetySetting
from sentence_transformers import SentenceTransformer, util
from vertexai.generative_models import GenerationConfig, GenerativeModel, HarmBlockThreshold, HarmCategory

from agentic_rl.base.log.loggers import Loggers
from agentic_rl.base.utils.globals import GCP_LOCATION, GCP_PROJECT_ID, GEMINI_MODEL, OAI_RM_MODEL

logger = Loggers(__name__).get_logger()


def compute_pass_at_k(results):
    import hashlib
    import json

    # Create a map to store correct answers per problem
    problem_correct_map: defaultdict[str, int] = defaultdict(int)
    problem_total_map: defaultdict[str, int] = defaultdict(int)

    # Count correct answers for each problem
    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = 1 if trajectory.reward > 0 else 0

        problem_correct_map[problem_hash] += is_correct
        problem_total_map[problem_hash] += 1

    # Calculate pass@1 and pass@16
    total_problems = len(problem_correct_map)
    pass_at_1 = sum(problem_correct_map.values()) / sum(problem_total_map.values())
    pass_at_k = sum(1 for problem, correct in problem_correct_map.items() if correct > 0) / total_problems

    logger.info("Total unique problems:", total_problems)
    logger.info("Average Pass@1 Accuracy:", pass_at_1)
    logger.info("Average Pass@k Accuracy:", pass_at_k)


def call_oai_rm_llm(
        prompt: str,
        system_prompt: str,
        n: int = 1,
        temperature: float = 1.0,
        model_id: str = OAI_RM_MODEL,
        retry_count: int = int(1e9),
) -> list[str]:
    client = openai.OpenAI()

    backoff = 1
    retry_count = int(retry_count)

    response = None
    for attempt in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                temperature=temperature,
                n=n,
            )
            break
        except Exception as e:
            if "429" in str(e):
                logger.info("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            else:
                logger.info("Exception: ", e)
                return []

    if response is None:
        return []
    if n == 1:
        content = response.choices[0].message.content
        return [content] if content is not None else []
    return [choice.message.content for choice in response.choices if choice.message.content is not None]


def call_gemini_llm(
        prompt: str,
        system_prompt: str,
        n: int = 1,
        temperature: float = 1.0,
        project_id: str = GCP_PROJECT_ID,
        location: str = GCP_LOCATION,
        model_id: str = GEMINI_MODEL,
        retry_count: int = int(1e9),
) -> list[str]:
    """
    Calls a Gemini LLM on Vertex AI to generate n responses at a given temperature.

    Args:
        prompt (str): The text prompt to send to the LLM.
        system_prompt (str): System instruction or system prompt to send to the model.
        n (int): Number of responses to generate.
        temperature (float): Sampling temperature.
        project_id (str): Your GCP project ID.
        location (str): The region to use (e.g., us-central1).
        model_id (str): The specific Gemini model resource name.
        retry_count (int): Number of times to retry on rate-limit errors.

    Returns:
        List[str]: A list of response texts from the Gemini model.
    """

    # Initialize the Vertex AI environment
    vertexai.init(project=project_id, location=location)

    # Define which harm categories to allow (or set thresholds).
    harm_categories = [
        HarmCategory.HARM_CATEGORY_UNSPECIFIED,
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
    ]

    # Instantiate the GenerativeModel
    model = GenerativeModel(
        model_name=model_id,
        system_instruction=[system_prompt],
    )

    # Add an exponential backoff for rate limit errors
    backoff = 1
    retry_count = int(retry_count)
    generation_config = GenerationConfig(
        temperature=temperature,
        candidate_count=n,
    )

    response = None
    for attempt in range(retry_count):
        try:
            # Request multiple candidates by specifying n (candidate_count)
            response = model.generate_content([prompt], generation_config=generation_config, safety_settings=[
                SafetySetting(category=h, threshold=HarmBlockThreshold.BLOCK_NONE) for h in harm_categories])
            # Once successful, break out of the retry loop
            break
        except Exception as e:
            # Retry if there's a rate-limit error (HTTP 429)
            if "429" in str(e):
                logger.warning("Retry due to rate limit: ", e)
                time.sleep(backoff)
                backoff = min(backoff * 2, 64)  # Exponential backoff up to 64s
                continue
            elif "403" in str(e):
                logger.error("NO ACCESS TO ENDPOINT", e)
                raise NotImplementedError from None
            else:
                logger.error("Exception: ", e)
                return []  # or raise an exception if desired

    # Collect the texts from all returned candidates
    # Depending on the library version, this might need to be adjusted
    # if the `response` shape is different

    try:
        # Keep this to check for errors in indexing.
        var = [candidate.text for candidate in response.candidates]
        if len(response.candidates) == 1:
            return response.candidates[0].text
        return [candidate.text for candidate in response.candidates]
    except Exception as e:
        logger.error("Error extracting text from response:", e)
        return []


class RAG:
    def __init__(self, docs: list[str], model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Args:
            docs (List[str]): A list of documents to encode.
            model (str): The SentenceTransformer model to use.
        """
        # Load the SentenceTransformer model
        self.model = SentenceTransformer(model)
        self.docs = docs
        # Compute embeddings
        self.embeddings = self.model.encode(docs, convert_to_tensor=True)

    def top_k(self, query, k=1):
        # Create embedding for the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Compute cosine similarity [1 x N]
        cos_scores = util.cos_sim(query_embedding, self.embeddings)[0]

        # Extract top_k indices
        top_results = torch.topk(cos_scores, k=k)

        # Prepare a list of (score, problem_text)
        results = []
        for score, idx in zip(top_results.values, top_results.indices, strict=False):
            results.append(
                {
                    "score": score,
                    "text": self.docs[int(idx)],
                    "idx": int(idx),
                }
            )
        return results


def strftime(timestamp):
    # Handling timestamp (supports floating-point numbers in seconds, such as 1723079445.123456)
    dt = datetime.datetime.fromtimestamp(timestamp)
    # Formatting includes microseconds
    formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return formatted_time


def _get_ip_by_ifname():
    """
    Obtain the IPv4 address through the interface name (such as eth0, en0)
    Return the IP string. If unsuccessful, return None.
    """
    try:
        # Execute the "ifconfig" command and capture the output
        ifname = os.environ.get("HCCL_SOCKET_IFNAME", 0)
        if ifname:
            output = subprocess.check_output(["ifconfig", ifname], stderr=subprocess.STDOUT).decode()
            # Regular expression matching for IPv4 addresses (excluding 127.0.0.1)
            matches = re.findall(r'inet (?:addr:)?((?:\d{1,3}\.){3}\d{1,3})', output)
            for ip in matches:
                if ip != "127.0.0.1":
                    return ip
        return None
    except subprocess.CalledProcessError:
        return None


def get_current_node_ip() -> str:
    try:
        # Create a UDP socket (only for obtaining interface information)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # Connect to an external address (without actual communication)
            s.connect(("8.8.8.8", 80))  # Google DNS 服务器
            local_ip = s.getsockname()[0]
    except Exception:
        local_ip = _get_ip_by_ifname()
        if not local_ip:
            # If it fails, revert to traversing the interface.
            local_ip = "127.0.0.1"
            hostname = socket.gethostname()
            for addr in socket.getaddrinfo(hostname, None):
                ip = addr[4][0]
                if not ip.startswith("::"):
                    local_ip = ip
                    break
    return local_ip


def get_cluster_info():
    # Ensure that the distributed environment has been initialized.
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized")

    world_size = dist.get_world_size()

    # Obtain the IP address of the current node
    ip_address = get_current_node_ip()

    # Collect all the IP addresses of all ranks
    ip_list = [None] * world_size
    dist.all_gather_object(ip_list, ip_address)
    return ip_list


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance
