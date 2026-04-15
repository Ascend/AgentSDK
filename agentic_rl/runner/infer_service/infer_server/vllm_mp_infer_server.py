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


# Standard library imports
import atexit
import json
import os
import signal
import subprocess
import threading
import time
import traceback
from typing import Dict, Union, Callable, Tuple, Any, Optional, List

# Third-party library imports
import requests
from openai import AsyncOpenAI

# Internal imports
from agentic_rl.base.log.loggers import Loggers
from agentic_rl.runner.infer_service.base_infer_server import BaseInferServer

logger = Loggers(__name__).get_logger()


# =============================================================================
# Start vLLM in MP mode on remote nodes (divided into Master and Slave nodes, using all cards on the node)
# =============================================================================

def print_log(out_buffer, key_word, key_event):
    idx = 0
    while True:
        if idx < len(out_buffer):
            line = out_buffer[idx].rstrip("\n")
            if key_word in line:
                key_event.set()
            logger.info(line)
            idx += 1
        else:
            time.sleep(0.01)


def start_cmd(cmd, out_buffer: list):
    server = None

    def cleanup():
        """Send SIGTERM signal to vLLM process group when parent process exits."""
        if server and server.poll() is None:
            try:
                pgid = os.getpgid(server.pid)
                logger.info(f"Parent process exiting, cleaning up child process group PGID - {pgid}...")
                os.killpg(pgid, signal.SIGTERM)
                server.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass  # Process is already dead or timeout, ignoring

    try:
        logger.info(f"{os.environ['ASCEND_RT_VISIBLE_DEVICES']=}")
        device_ids = sorted([int(d) for d in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',') if d.strip().isdigit()])
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ",".join(map(str, device_ids))
        os.environ['VLLM_SERVER_DEV_MODE'] = "1"
        server = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr to stdout
            bufsize=1,  # Line buffering
            text=False,  # Direct str instead of bytes
            start_new_session=True
        )
        atexit.register(cleanup)
        logger.info(f"Child process started (PID: {server.pid}), output is being written to list in real-time...")

        for line_bytes in iter(server.stdout.readline, b''):
            line = line_bytes.decode('utf-8', errors='replace')
            out_buffer.append(line)

        server.stdout.close()
        server.wait()  # Wait for child process to exit completely
        logger.info("Child process has exited.")

    except KeyboardInterrupt:
        logger.error("User interrupted.")
    except Exception as e:
        logger.error(f"Failed to start: {e}")


def map_dict_to_cli_args(
        params: Dict[str, Any],
        initial_args: Optional[List[str]] = None
) -> List[str]:
    """
    Convert parameters in a dictionary (keys in snake_case) to a list of command line arguments (kebab-case).

    Mapping rules:
    1.  Convert underscores '_' in dictionary keys to hyphens '-'.
    2.  If the value is boolean True, only add the parameter name (i.e., --param-name).
    3.  If the value is boolean False, skip this parameter.
    4.  If the value is non-None, add both parameter name and value (i.e., --param-name value).

    Args:
        params: Dictionary containing parameters, keys use snake_case (e.g., 'data_parallel_size').
        initial_args: Optional initial argument list, such as model path.

    Returns:
        A list of strings in command line format (e.g., ['--data-parallel-size', '2']).
    """

    cli_args = initial_args.copy() if initial_args else []

    for key, value in params.items():
        # Rule 1: Convert to kebab-case
        cli_key = "--" + key.replace('_', '-')

        if value is True:
            # Rule 2: Boolean True, only add parameter name
            cli_args.append(cli_key)

        elif value is False or value is None:
            # Rule 3: Boolean False or None, skip this parameter
            continue

        else:
            # Rule 4: Other non-None values, add parameter name and value
            cli_args.append(cli_key)
            # Ensure value is string type
            cli_args.append(str(value))

    return cli_args

def start_slave(index, master_addr, kwargs):
    # ================= Modify special variables and addresses =================
    model = kwargs.pop('model')
    kwargs['data_parallel_address'] = master_addr
    kwargs['data_parallel_start_rank'] = index * kwargs['data_parallel_size_local']
    kwargs["worker_extension_cls"] = ("agentic_rl.runner.infer_adapter.vllm." 
                                      "extension.custom_worker_extensions.CustomWorkerExtensions")

    # ================= Simulate complete command line arguments list =================
    cli_args = map_dict_to_cli_args(kwargs, ["vllm", "serve", model, "--headless"])
    logger.info(f"Start slave args: {cli_args}.")

    out_buffer = []
    ready_word, ready_event = 'Application startup complete.', threading.Event()
    threading.Thread(target=start_cmd, args=(cli_args, out_buffer,)).start()
    threading.Thread(target=print_log, args=(out_buffer, ready_word, ready_event)).start()


def start_master(master_addr, kwargs):
    # ================= Modify special variables and addresses =================
    model = kwargs.pop('model')
    kwargs['data_parallel_address'] = master_addr
    kwargs["worker_extension_cls"] = ("agentic_rl.runner.infer_adapter.vllm." 
                                      "extension.custom_worker_extensions.CustomWorkerExtensions")

    # ================= Simulate complete command line arguments list =================
    cli_args = map_dict_to_cli_args(kwargs, ["vllm", "serve", model])
    logger.info(f"Start master args: {cli_args}.")

    out_buffer = []
    ready_word, ready_event = 'Application startup complete.', threading.Event()
    threading.Thread(target=start_cmd, args=(cli_args, out_buffer,)).start()
    threading.Thread(target=print_log, args=(out_buffer, ready_word, ready_event)).start()
    ready_event.wait()


# =============================================================================
# Start vLLM remote processes in MP mode on remote nodes (one per node)
# =============================================================================

class RemoteMPVLLMInferServer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.engine = None
        self.openai_serving_chat = None

    async def init_server(self, index, master_addr, model_name, **kwargs):
        await self.init_environment()

        local_addr = await self.get_local_addr()
        is_master = local_addr == master_addr

        if is_master:
            logger.info(f"Running on master node: {local_addr}")
            start_master(master_addr=master_addr, kwargs=kwargs)
        else:
            logger.info(f"Running on slave node: {local_addr}")
            start_slave(index=index, master_addr=master_addr, kwargs=kwargs)
        logger.info(f"RemoteMPVLLMInferServer {is_master=} is init finished.")

    async def init_environment(self):
        # 1. Dynamically get IP and network interface card name
        local_addr = await self.get_local_addr()
        nic_name = await self.get_nic_name(local_addr)

        logger.info(f"Configuring Environment: IP={local_addr}, NIC={nic_name}")

        # 2. Set dynamic environment variables (dependent on obtained values)
        os.environ['HCCL_IF_IP'] = local_addr

        # The following usually need to be bound to specific network interface names
        os.environ['GLOO_SOCKET_IFNAME'] = nic_name
        os.environ['TP_SOCKET_IFNAME'] = nic_name
        os.environ['HCCL_SOCKET_IFNAME'] = nic_name

        # 3. Set static environment variables
        # Note: os.environ values must be strings (str)
        os.environ['OMP_PROC_BIND'] = 'false'  # corresponds to false
        os.environ['OMP_NUM_THREADS'] = '100'  # corresponds to 100
        os.environ['VLLM_USE_V1'] = '1'  # corresponds to 1
        os.environ['HCCL_BUFFSIZE'] = '1024'  # corresponds to 1024

    @classmethod
    async def get_nic_name(cls, local_ip) -> str:
        import psutil
        interfaces = psutil.net_if_addrs()

        for nic_name, snics in interfaces.items():
            for snic in snics:
                # AF_INET represents IPv4
                import socket
                if snic.family == socket.AF_INET:
                    if snic.address == local_ip:
                        return nic_name

        return "Unknown"

    @classmethod
    async def get_local_addr(cls) -> str | None:
        s = None
        try:
            # Create a UDP socket
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Connect to a public DNS address (Google DNS) on any port
            # Note: This step does not actually send data, no handshake needed, so it's very fast and doesn't consume traffic
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
        except Exception as e:
            # If connection fails (e.g., completely offline), fallback to localhost
            traceback.print_exc()
            logger.error(f"get_local_ip failed: {e}.")
            ip = '127.0.0.1'
        finally:
            if s:
                s.close()

        return ip


# =============================================================================
# Main process entry for starting vLLM in MP mode on remote nodes
# =============================================================================

class VLLMMPInferServer(BaseInferServer):
    def __init__(self, model_name, **kwargs):
        if "distributed_executor_backend" in kwargs and kwargs["distributed_executor_backend"] != "mp":
            raise RuntimeError("distributed_executor_backend must be mp.")

        import ray
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        placement_group = ray.util.get_current_placement_group()

        if placement_group is None:
            raise RuntimeError("Current actor is not running in a Placement Group.")
        
        bundles = placement_group.bundle_specs
        logger.info(f"Bundle specs: {bundles}")

        #================================================
        # 1. Create Server and set affinity
        #================================================ 
        self.master_server = ray.remote(RemoteMPVLLMInferServer).options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=0,  # Bind to index 0
            ),
            num_cpus=0,
            resources=bundles[0]
        ).remote()

        self.slave_servers = [
            ray.remote(RemoteMPVLLMInferServer).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement_group,
                    placement_group_bundle_index=i,  # Bind to index i
                ),
                num_cpus=0,
                resources=bundles[i]
            ).remote()
            for i in range(1, len(bundles))
        ]

        #================================================
        # 2. Initialize all Servers
        #================================================ 
        master_addr = ray.get(self.master_server.get_local_addr.remote())
        futures = [self.master_server.init_server.remote(
            index=0, master_addr=master_addr, model_name=model_name, **kwargs
        )] + [server.init_server.remote(index=i + 1, master_addr=master_addr, model_name=model_name, **kwargs)
              for i, server in enumerate(self.slave_servers)]
        for f in futures:
            ray.get(f)

        self.vllm_server_addr = f"http://{master_addr}:{kwargs['port']}"
        self.client = AsyncOpenAI(
            base_url=self.vllm_server_addr + '/v1/',
            api_key='EMPTY'
        )
        self.model = kwargs['model']
        logger.info(f"VLLMMPInferServer init done: {self.model=}.")

    async def completions(self, request_data: Dict):
        request_data['logprobs'] = 1
        request_data['extra_body'] = {"return_token_ids": True}
        # Remove 'stream' key to ensure non-streaming request
        request_data.pop("stream", None)
        request_data['model'] = self.model

        if 'extra_headers' in request_data:
            request_data.pop('extra_headers')

        # Call OpenAI SDK asynchronous method
        completion = await self.client.completions.create(**request_data)

        # Return dictionary representation of Pydantic model
        return completion.model_dump()

    async def chat_completions(self, request_data: Dict):
        request_data['logprobs'] = 1
        request_data['extra_body'] = {"return_token_ids": True}
        # Remove 'stream' key to ensure non-streaming request
        request_data.pop("stream", None)
        request_data['model'] = self.model

        if 'extra_headers' in request_data:
            request_data.pop('extra_headers')

        # Call OpenAI SDK asynchronous method
        completion = await self.client.chat.completions.create(
            **request_data
        )

        # Return dictionary representation of Pydantic model
        return completion.model_dump()

    async def stream_chat_completions(self, request_data: Dict):
        # Ensure stream key is set to True to trigger streaming response
        request_data["stream"] = True
        request_data['model'] = self.model

        # Call OpenAI SDK asynchronous method
        stream = await self.client.chat.completions.create(
            **request_data
        )

        # Asynchronously iterate through the stream and generate dictionary representation of each chunk
        async for chunk in stream:
            json_string = json.dumps(chunk.model_dump(), ensure_ascii=False)
            yield json_string

    async def collective_rpc(
            self,
            method: Union[str, Callable],
            timeout: Optional[float] = 600,
            args: Tuple = (),
            kwargs: Optional[Dict[str, Any]] = None,
    ) -> List:
        # Construct request body
        payload = {
            "method": method,
            "args": args if args is not None else [],
            "kwargs": kwargs if kwargs is not None else {},
        }

        collective_rpc_url = self.vllm_server_addr + '/collective_rpc'
        try:
            logger.info(f"collective_rpc_url={collective_rpc_url}, payload={payload}")
            response = requests.post(
                collective_rpc_url,
                json=payload,
                timeout=timeout + 5  # The timeout for the `requests` library can be set slightly longer than the RPC timeout
            )

            response.raise_for_status()
            if response.status_code == 200:
                # Parse JSON response
                result = response.json()
                logger.info(f"collective_rpc_url={collective_rpc_url}, result={result}")
                return result
            return None
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise e

    async def get_workload(self):
        pass

    async def cancel_requests(self, *args, **kwargs):
        pass
