#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
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


import argparse
import asyncio
import functools
import heapq
import httpx
import os
import sys
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import List
from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass


class ServerState:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.url = f'http://{host}:{port}/v1'
        self.client = httpx.AsyncClient(
            timeout=None,
            base_url=self.url,
            limits=httpx.Limits(
                max_connections=100000,
                max_keepalive_connections=100000,
            ))
        self.active_tokens = 0
        self.active_kv_cache = 0
        self.active_requests = 0
        self.aborted_requests = set()


class ProxyState:
    def __init__(self, prefiller_instances, decoder_instances):
        self.prefillers: List[ServerState] = [
            ServerState(h, p) for h, p in prefiller_instances
        ]
        self.decoders: List[ServerState] = [
            ServerState(h, p) for h, p in decoder_instances
        ]
        self.req_to_prefiller = {}
        self.req_id_lock = asyncio.Lock()

        self.prefiller_heap = [(0, i, server)
                                for i, server in enumerate(self.prefillers)]
        self.decoder_heap = [(0, i, server)
                              for i, server in enumerate(self.decoders)]
        heapq.heapify(self.prefiller_heap)
        heapq.heapify(self.decoder_heap)

    def _update_prefiller_priority(self, server_idx: int):
        server = self.prefillers[server_idx]
        priority = server.active_tokens + server.active_kv_cache * 0.3
        self.prefiller_heap = [(p, i, s) for p, i , s in self.prefiller_heap
                                if i != server_idx]
        heapq.heappush(self.prefiller_heap, (priority, server_idx, server))

    def _update_decoder_priority(self, server_idx: int):
        server = self.decoders[server_idx]
        priority = server.active_tokens
        self.decoder_heap = [(p, i, s) for p, i , s in self.decoder_heap
                                if i != server_idx]
        heapq.heappush(self.decoder_heap, (priority, server_idx, server))
    
    def abort_prefiller_request(self, server_idx: int, request_id):
        self.prefillers[server_idx].aborted_requests.add(request_id)
    
    def aquire_aborted_prefiller_requests(self, server_idx: int):
        aborted_requests = self.prefillers[server_idx].aborted_requests.copy()
        self.prefillers[server_idx].aborted_requests.clear()
        return aborted_requests
    
    async def next_req_id(self):
        async with self.req_id_lock:
            return str(uuid.uuid4())
    
    def select_prefiller(self, token_count):
        if not self.prefiller_heap:
            raise RuntimeError("No available prefiller servers.")
        
        priority, chosen, server = heapq.heappop(self.prefiller_heap)

        self.prefillers[chosen].active_tokens += token_count
        self.prefillers[chosen].active_kv_cache += token_count

        self._update_prefiller_priority(chosen)

        return chosen

    def release_prefiller(self, idx, token_count):
        self.prefillers[idx].active_tokens -= token_count
        self._update_prefiller_priority(idx)

    def release_prefiller_kv(self, idx, token_count):
        if self.prefillers[idx].active_kv_cache > 0:
            self.prefillers[idx].active_kv_cache -= token_count
        self._update_prefiller_priority(idx)
    
    def select_decoder(self, token_count):
        if not self.decoder_heap:
            raise RuntimeError("No available decoder servers.")
        
        priority, chosen, server = heapq.heappop(self.decoder_heap)

        self.decoders[chosen].active_tokens += token_count

        self._update_decoder_priority(chosen)

        return chosen
    
    def release_decoder(self, idx, token_count):
        self.decoders[idx].active_tokens -= token_count
        self._update_decoder_priority(idx)

    def calculate_prefill_scores(self, request_length: int) -> float:
        length_score = request_length / 4.0
        input_score = length_score * 0.0345 + 120.0745
        return input_score
    
    def calculate_decode_scores(self, request_length: int) -> float:
        return request_length


proxy_state = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--prefiller-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--prefiller-ports", type=int, nargs="+", default=[8001])
    parser.add_argument("--decoder-hosts", type=str, nargs="+", default=["localhost"])
    parser.add_argument("--decoder-ports", type=int, nargs="+", default=[8002])
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
    parser.add_argument("--retry-delay", type=float, default=0.001, help="Base delay (seconds) for exponential backoff retries")
    args = parser.parse_args()
    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError("Number of prefiller hosts must match number of prefiller ports.")
    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports.")
    args.prefiller_instances = list(zip(args.prefiller_hosts, args.prefiller_ports))
    args.decoder_instances = list(zip(args.decoder_hosts, args.decoder_ports))
    
    return args


@asynccontextmanager
async def lifespan(app: FastAPI):
    global proxy_state
    proxy_state = ProxyState(global_args.prefiller_instances, global_args.decoder_instances)
    
    print(f"Initialized {len(proxy_state.prefillers)} prefill clients and {len(proxy_state.decoders)} decode clients.")
    yield
    for p in proxy_state.prefillers:
        await p.client.aclose()
    for d in proxy_state.decoders:
        await d.client.aclose()


async def listen_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message['type'] == 'http.disconnect':
            break


def with_cancellation(handler_func):
    @functools.wraps(handler_func)
    async def wrapper(*args, **kwargs):
        request = kwargs["request"]
        handler_task = asyncio.create_task(handler_func(*args, **kwargs))
        cancellation_task = asyncio.create_task(listen_for_disconnect(request))
        done, pending = await asyncio.wait([handler_task, cancellation_task],
                                            return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if handler_task in done:
            return handler_task.result()
        return None
    
    return wrapper


app = FastAPI(lifespan=lifespan)


async def send_request_to_service(client: httpx.AsyncClient, 
                                 prefiller_id: int,
                                 endpoint: str,
                                 req_data: dict,
                                 request_id: str,
                                 max_retries: int = 3,
                                 base_delay: float = 0.2):
    aborted_requests = proxy_state.aquire_aborted_prefiller_requests(prefiller_id)
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
        "aborted_request": list(aborted_requests),
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    req_data["min_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {
        "Authorization": f"Bearer {os.environ.get("OPENAI_API_KEY")}",
        "X-Request-Id": request_id
    }
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = await client.post(endpoint, 
                                         json=req_data, 
                                         headers=headers)
            response.raise_for_status()
            return response
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            logger.warning(
                f"Attempt {attempt} failed for {endpoint}: {str(e)}")
            last_exc = e
            if attempt < max_retries:
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for {endpoint}")
                raise last_exc


async def stream_service_response_with_retry(client: httpx.AsyncClient,
                                             endpoint: str,
                                             req_data: dict,
                                             request_id: str,
                                             max_retries: int = 3,
                                             base_delay: float = 0.2):
    headers = {
        "Authorization": f"Bearer {os.environ.get("OPENAI_API_KEY")}",
        "X-Request-Id": request_id
    }
    for attempt in range(1, max_retries + 1):
        try:
            async with client.stream("POST",
                                     endpoint, 
                                     json=req_data, 
                                     headers=headers) as response:
                response.raise_for_status()
                first_chunk_sent = False
                async for chunk in response.aiter_bytes():
                    first_chunk_sent = True
                    yield chunk
                return
        except (httpx.RequestError, httpx.HTTPStatusError) as e:
            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
            else:
                logger.error(
                    f"All {max_retries} attempts failed for streaming {endpoint}")
                raise e
        except Exception as e:
            if 'first_chunk_sent' in locals() and first_chunk_sent:
                logger.error(
                    f"Streaming to client interrupted after response started: {str(e)}")
                return
            else:
                if attempt < max_retries:
                    logger.warning(
                        f"Attempt {attempt} failed for streaming {endpoint}: {str(e)}")
                    await asyncio.sleep(base_delay * (2 ** (attempt - 1)))
                else:
                    logger.error(
                        f"All {max_retries} attempts failed for streaming {endpoint}")
                    raise e


async def _handle_completions(api: str, request: Request):
    try:
        req_data = await request.json()
        req_body = await request.body()
        request_length = len(req_body)
        prefiller_score = proxy_state.calculate_prefill_scores(request_length)
        logger.debug(f"Request length: {request_length}, Prefiller score: {prefiller_score}")
        request_id = await proxy_state.next_req_id()
        prefiller_idx = proxy_state.select_prefiller(prefiller_score)
        prefiller = proxy_state.prefillers[prefiller_idx]
        response = await send_request_to_service(
            prefiller.client,
            prefiller_idx,
            api,
            req_data,
            request_id,
            max_retries=global_args.max_retries,
            base_delay=global_args.retry_delay,
        )
        proxy_state.release_prefiller(prefiller_idx, prefiller_score)
        response_json = response.json()
        kv_transfer_params = response_json.get("kv_transfer_params", {})
        if kv_transfer_params:
            req_data['kv_transfer_params'] = kv_transfer_params
        decoder_score = proxy_state.calculate_decode_scores(request_length)
        logger.debug("Decoder score: %f", decoder_score)
        decoder_idx = proxy_state.select_decoder(decoder_score)
        decoder = proxy_state.decoders[decoder_idx]
        logger.debug("Using %s %s", prefiller.url, decoder.url)
        released_kv = False

        async def generate_stream():
            nonlocal released_kv
            try:
                async for chunk in stream_service_response_with_retry(
                        decoder.client,
                        api,
                        req_data,
                        request_id=request_id,
                        max_retries=global_args.max_retries,
                        base_delay=global_args.retry_delay):
                    if not released_kv and chunk:
                        proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
                        released_kv = True
                    yield chunk
            except Exception as e:
                logger.error(
                    f"Error during streaming from decoder {decoder.url}: {str(e)} the aborted request {request_id} will be routing to the target prefiller when new request is ready to dispatch to it"
                )
                proxy_state.abort_prefiller_request(prefiller_idx, request_id)
                proxy_state.release_prefiller_kv(prefiller_idx, prefiller_score)
            
            proxy_state.release_decoder(decoder_idx, decoder_score)
        
        return StreamingResponse(generate_stream(), media_type="application/json")
    except Exception as e:
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server"
             f" - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
@with_cancellation
async def handle_completions(request: Request):
    return await _handle_completions("/completions", request)


@app.post("/v1/chat/completions")
@with_cancellation
async def handle_chat_completions(request: Request):
    return await _handle_completions("/chat/completions", request)


@app.get("/healthcheck")
async def healthcheck():
    return {
        "status": "ok",
        "prefill_instances": len(proxy_state.prefillers),
        "decode_instances": len(proxy_state.decoders)
    }


if __name__=='__main__':
    global global_args
    global_args = parse_args()
    import uvicorn
    uvicorn.run(app, host=global_args.host, port=global_args.port)