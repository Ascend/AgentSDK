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

import threading
from typing import List

import llm_datadist
import msgspec
import zmq
from llm_datadist import BlocksCacheKey, CacheDesc, LLMConfig, LLMDataDist, LLMException, LLMRole
from vllm.utils import get_ip, logger
import vllm_ascend.envs as envs_ascend
from vllm_ascend.distributed.llmdatadist_c_mgr_connector import (
    LLMDataDistCMgrEvent, LLMDataDistCMgrAgentMetadata,
    LLMDataDistCMgrConnectorWorker, zmq_ctx)


def listen_for_agent_metadata_req_patch(self, event: threading.Event):
    if self.local_agent_metadata is None:
        raise RuntimeError("Local agent metadata must be initialized before listening for requests")
    
    port = (envs_ascend.VLLM_ASCEND_LLMDD_RPC_PORT + self.local_dp_rank * self.tp_size + self.tp_rank
            if self.local_dp_rank is not None
            else envs_ascend.VLLM_ASCEND_LLMDD_RPC_PORT + self.tp_size + self.tp_rank)
    url = f"tcp://{envs_ascend.VLLM_ASCEND_LLMDD_RPC_IP}:{port}"
    msg_encoder = msgspec.msgpack.Encoder()
    msg_decoder = msgspec.msgpack.Decoder()
    msg_to_send = msg_encoder.encode(self.local_agent_metadata)
    logger.debug(f"Start to listen to address: {url}")
    logger.debug(f"The local agent metadata have {len(msg_to_send)} bytes here")
    logger.info(
        f"LLMDataDistCMgrConnectorWorker: Cluster {self.local_agent_metadata.cluster_id} "
        f"start to listen request from peers"
    )
    
    with zmq_ctx(zmq.ROUTER, url) as sock:  # type: ignore[attr-defined]
        event.set()
        while True:
            identity, _, msg = sock.recv_multipart()
            event_msg, decode_msg = msg_decoder.decode(msg)
            event_msg = LLMDataDistCMgrEvent(event_msg)
            if event_msg == LLMDataDistCMgrEvent.ReqForMetadata:
                if "cluster_id" in decode_msg:
                    decode_msg = LLMDataDistCMgrAgentMetadata(**decode_msg)
                    logger.info(
                        f"LLMDataDistCMgrConnectorWorker: Receive message from cluster {decode_msg.cluster_id}"
                    )
                    sock.send_multipart((identity, b"", msg_to_send))
                    self.add_remote_agent(decode_msg)
                else:
                    logger.warning(
                        f"LLMDataDistCMgrConnectorWorker: receiving unrecognized data {decode_msg}"
                    )
            elif event_msg == LLMDataDistCMgrEvent.ReqForFinished:
                finished_req_id = decode_msg[0]
                with self.thread_lock:
                    logger.debug(
                        f"LLMDataDistCMgrConnectorWorker: Receiving request {finished_req_id} finished"
                    )
                    self.finished_reqs.add(finished_req_id)
                sock.send_multipart((identity, b"", b"receiving decode finished"))
            else:
                raise RuntimeError(
                    f"LLMDataDistCMgrConnectorWorker: Receiving unexpected request event {event_msg} from remote !"
                )


def send_finish_to_remote(self, host: str, ports: List[int], request_id: str) -> None:
    for port in ports:
        url = f"tcp://{host}:{port}"
        logger.debug(f"Sending finished to remote: {url}")
        msg_encoder = msgspec.msgpack.Encoder()
        msg_send = msg_encoder.encode([LLMDataDistCMgrEvent.ReqForFinished, [request_id]])
        with zmq_ctx(zmq.REQ, url) as sock:  # type: ignore[attr-defined]
            try:
                sock.send(msg_send)
                logger.debug(f"Request id {request_id} finished message send to remote {url}")
                _ = sock.recv()
            except Exception as e:
                logger.error(f"Failed to send reqest_id {request_id} to prefill: {e}")


def _read_blocks_patch(
    self,
    local_block_ids: List[int],
    remote_block_ids: List[int],
    remote_ip: str,
    remote_port: int,
    remote_engine_id: str,
    request_id: str,
    remote_tp_size: str,
):
    tp_offset = self.tp_rank % int(remote_tp_size)
    remote_cluster_id = self.connect_to_remote_agent(remote_ip, remote_port + tp_offset)
    num_local_blocks = len(local_block_ids)
    if num_local_blocks == 0:
        return
    
    num_remote_blocks = len(remote_block_ids)
    if num_local_blocks > num_remote_blocks:
        raise RuntimeError(
            f"Number of local blocks ({num_local_blocks}) cannot exceed "
            f"number of remote blocks ({num_remote_blocks})"
        )
    
    if num_local_blocks < num_remote_blocks:
        remote_block_ids = remote_block_ids[-num_local_blocks:]

    logger.info(f"remote cluster id is: {remote_cluster_id}")
    if self.use_mla:
        remote_cache_key_k_normed = BlocksCacheKey(cluster_id=remote_cluster_id, model_id=0)
        remote_cache_key_k_pe = BlocksCacheKey(cluster_id=remote_cluster_id, model_id=1)
        logger.info("Try pull blocks from remote server")
        try:
            self.cache_manager.pull_blocks(
                remote_cache_key_k_normed,
                self.cache[0],  # type: ignore[has-type]
                remote_block_ids,
                local_block_ids)
            self.cache_manager.pull_blocks(
                remote_cache_key_k_pe,
                self.cache[1],  # type: ignore[has-type]
                remote_block_ids,
                local_block_ids)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"LLMDataDistCMgrConnectorWorker: Passing unexpected parameter to pull_blocks "
                f"remote_cache_key: {remote_cache_key_k_normed} {remote_cache_key_k_pe}, "
                f"cache: {self.cache}, local_block_ids: {local_block_ids}, "
                f"remote_block_ids: {remote_block_ids}"  # type: ignore[has-type]
            )
        except LLMException:
            raise RuntimeError(
                "LLMDataDistCMgrConnectorWorker: Timeout during pull_blocks, "
                "you can try to increase the sync_kv_timeout config or checking your connect status"
            )
    else:
        remote_cache_key = BlocksCacheKey(cluster_id=remote_cluster_id)
        logger.info("Try pull blocks from remote server")
        try:
            self.cache_manager.pull_blocks(
                remote_cache_key,
                self.cache,  # type: ignore[has-type]
                remote_block_ids,
                local_block_ids)
        except (TypeError, ValueError):
            raise RuntimeError(
                f"LLMDataDistCMgrConnectorWorker: Passing unexpected parameter to pull_blocks "
                f"remote_cache_key: {remote_cache_key}, cache: {self.cache}, "
                f"local_block_ids: {local_block_ids}, remote_block_ids: {remote_block_ids}"  # type: ignore[has-type]
            )
        except LLMException:
            raise RuntimeError(
                "LLMDataDistCMgrConnectorWorker: Timeout during pull_blocks, "
                "you can try to increase the sync_kv_timeout config or checking your connect status"
            )
    
    remote_ports = list(
        range(remote_port + self.tp_rank,
              remote_port + int(remote_tp_size), self.tp_size))
    self.send_finish_to_remote(remote_ip, remote_ports, request_id)
    with self.thread_lock:
        self.finished_reqs.add(request_id)


LLMDataDistCMgrConnectorWorker.listen_for_agent_metadata_req = listen_for_agent_metadata_req_patch
LLMDataDistCMgrConnectorWorker._read_blocks = _read_blocks_patch
