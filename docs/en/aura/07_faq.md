# FAQ<a name="faq_main"></a>

## Network Unavailable<a name="faq_001"></a>

**Symptom<a name="faq_001_phenomenon"></a>**

After starting Aura, the following error message is displayed.

```text
eth0: error fetching interface information: Device not found
[ERROR] [ 26-05-21 10:47:56 ] can not get IP from eth0
```

**Cause Analysis<a name="faq_001_analysis"></a>**

By default, the system obtains the local IP address through the `DEFAULT_SOCKET_IFNAME` environment variable, which is set to `eth0` by default. The error occurs because the network interface named `eth0` cannot be found using `ifconfig`, preventing the system from obtaining the correct local IP address.

**Solution<a name="faq_001_solution"></a>**

Check the network configuration in your container and change the value of the `DEFAULT_SOCKET_IFNAME` environment variable to the name of a network interface that exists in the current environment and has the correct local IP address, such as `br0`, `eth0`, or `enp189s0f0`.

The following provides an example.

1. Run the `ifconfig` command to view the network configuration.

    ```shell
    ifconfig
    ```

2. Assume that the following information is displayed (partial output).

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. Assume that the local IP address is `192.168.0.1`. The network interface associated with the local IP address is `enp189s0f0`. Therefore, run the following command.

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

## Line Endings Not Recognized<a name="faq_002"></a>

**Symptom<a name="faq_002_phenomenon"></a>**

After starting Aura, the following error message is displayed.

```text
$'\r': command not found
```

**Cause Analysis<a name="faq_002_analysis"></a>**

Shell scripts may use different line ending formats on different operating systems, resulting in parsing errors.

**Solution<a name="faq_002_solution"></a>**

Convert all shell scripts and configuration files to Unix format.

```shell
# Convert all shell scripts to Unix format
find /path/to/AgentSDK -type f -name "*.sh" -exec dos2unix {} +
```

## IP Address or Port Already in Use<a name="faq_003"></a>

**Symptom<a name="faq_003_phenomenon"></a>**

After starting Aura, the following error messages are displayed.

```text
RuntimeError: createHCCLCommOrigin:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:2314 HCCL function error: HcclGetRootInfo(&hcclID), error code is 7
ERR02200 DIST call hccl api failed
Failed to bind the IP port. Reason: The IP address and port have been bound already.
```

**Cause Analysis<a name="faq_003_analysis"></a>**

The HCCL port is occupied by another process, resulting in an HCCL error indicating that the IP address and port have already been bound.

**Solution<a name="faq_003_solution"></a>**

Set the following environment variables to change the HCCL ports.

```shell
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60100
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
```

## UID Mismatch<a name="faq_004"></a>

**Symptom<a name="faq_004_phenomenon"></a>**

In hybrid mode, the following error message is displayed during the final validation stage after training and inference are completed.

```text
AssertionError: 'uid' in tensor_dict1 and tensor_dict2 are not same object..
```

**Cause Analysis<a name="faq_004_analysis"></a>**

The `test_freq` parameter is incorrectly configured in the YAML file.

**Solution<a name="faq_004_solution"></a>**

Set the `test_freq` parameter in the YAML file to `-1` to disable the validation stage.

## Some Fields Missing from the `utils` File When Running Triton<a name="faq_005"></a>

**Symptom<a name="faq_005_phenomenon"></a>**

After starting the script, an error is reported in `triton/backends/ascend/utils.py`, indicating that `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` cannot be found.

```text
/tmp/tmpaxgs3vq5/npu_utils.cpp:324:3: error: could not convert '{{"LOW_POWER_TIMEOUT", RT_LIMIT_TYPE_LOW_POWER_TIMEOUT}, {"WARP_STACK_SIZE", <expression error>}, {"DVG_WARP_STACK_SIZE", RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE}, {"STACK_SIZE", RT_LIMIT_TYPE_STACK_SIZE}}' from '<brace-enclosed initializer list>' to 'const std::unordered_map<std::__cxx11::basic_string<char>, tagRtLimitType>' [repeated 3x across cluster]
```

**Cause Analysis<a name="faq_005_analysis"></a>**

This file exists in CANN 8.5.1 but has been removed in CANN 9.0.0. You can run the following command to search for the file.

```text
grep -R RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE /usr/local/Ascend/ascend-toolkit/latest/include/experiment/runtime/runtime
```

This issue exists in triton-ascend 3.2.0 and can be fixed by applying the following patches.

* [triton-ascend_driver.patch](../../../aura/third_party/patch/triton-ascend_driver.patch)
* [triton-ascend_npu_utils.patch](../../../aura/third_party/patch/triton-ascend_npu_utils.patch)

For the corresponding triton-ascend PR, see: <https://gitcode.com/Ascend/triton-ascend/pull/1525/diffs>

**Solution<a name="faq_005_solution"></a>**

You can run the provided script to apply the patches automatically.

```bash
bash /home/work/AgentSDK/docker/aura/patch/patch_triton_ascend.sh
```

Alternatively, you can manually apply the patches to the relevant parts of triton-ascend 3.2.0.

## `pkg_resources` Module Missing<a name="faq_006"></a>

**Symptom<a name="faq_006_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause Analysis<a name="faq_006_analysis"></a>**

A newer version of setuptools is installed, and the module has been removed from newer versions.

**Solution<a name="faq_006_solution"></a>**

Install setuptools 80.10.2.

## Unexpected `strict` Field in the Configuration File<a name="faq_007"></a>

**Symptom<a name="faq_007_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
TypeError("CheckpointConfig.__init__() got an unexpected keyword argument 'strict'")
```

**Cause Analysis<a name="faq_007_analysis"></a>**

The configuration file reads `verl` from the root directory. The `verl` version in the root directory is incorrect. Switch to the specified commit.

**Solution<a name="faq_007_solution"></a>**

Switch the `verl` version in the root directory to the specified commit.

```text
cd /
git clone https://github.com/verl-project/verl.git
cd verl
git checkout e9972368aa6a6078eacd7f0678bdfdd0196ce7b5
```

## `fused_moe.runner` Module Missing from vLLM<a name="faq_008"></a>

**Symptom<a name="faq_008_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
ModuleNotFoundError: No module named 'vllm.model_executor.layers.fused_moe.runner'
```

**Cause Analysis<a name="faq_008_analysis"></a>**

vLLM 0.16.0rc1 does not contain the `runner` directory. The directory was added in vLLM 0.16.1rc0. It is recommended that you switch directly to the specified commit.

For the version mapping between vLLM and vLLM-Ascend, see [vLLM-Ascend Versioning Policy](https://docs.vllm.com.cn/projects/ascend/en/latest/community/versioning_policy.html).

**Solution<a name="faq_008_solution"></a>**

Switch vLLM to the corresponding version.

```text
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout 4034c3d32
VLLM_TARGET_DEVICE=empty pip install -v -e .

git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout fe4cad24e
export COMPILE_CUSTOM_KERNELS=1 && pip install -v -e .
```

## `libtorch_cuda.so` File Missing<a name="faq_009"></a>

**Symptom<a name="faq_009_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory.
```

**Cause Analysis<a name="faq_009_analysis"></a>**

A CUDA version of PyTorch is being used. On the Ascend platform, use either TorchNPU or the CPU version of PyTorch. Therefore, uninstall the current PyTorch package and install the CPU version of PyTorch.

**Solution<a name="faq_009_solution"></a>**

Install the CPU version of PyTorch.

```text
pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

## `triton.language.target_info` Module Missing from Triton<a name="faq_010"></a>

**Symptom<a name="faq_010_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
Failed to import Triton kernels. Please make sure your triton version is compatible. Error: No module named 'triton.language.target_info'
```

**Cause Analysis<a name="faq_010_analysis"></a>**

This occurs because vLLM checks whether the current environment is a CUDA environment. If it is not a CUDA environment, vLLM logs an error and then proceeds with the Ascend vLLM workflow.

**Solution<a name="faq_010_solution"></a>**

This is part of the internal vLLM code logic and does not require a fix. The message is only informational and does not interrupt the main workflow. It is recommended that you investigate other possible causes of the error.

## `moe_distribute_comm_ctx.h` File Missing When Compiling and Installing vLLM-Ascend<a name="faq_011"></a>

**Symptom<a name="faq_011_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
fatal error: 'moe_distribute_comm_ctx.h' file not found
```

**Cause Analysis<a name="faq_011_analysis"></a>**

On A3 servers, vLLM-Ascend does not include the required header file path when compiling operators using CMake. You must configure the environment variable to add this path to the CMake search paths.

**Solution<a name="faq_011_solution"></a>**

Add the following environment variable to specify the header file path to be included when compiling operators with CMake.

```text
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel:${CPLUS_INCLUDE_PATH}
```

## HTTP Connection Failed<a name="faq_012"></a>

**Symptom<a name="faq_012_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
httpcore.ConnectError: All connection attempts failed
```

**Cause Analysis<a name="faq_012_analysis"></a>**

An HTTP proxy is configured in the environment, causing the network connection to fail when `InferExecutor` calls the vLLM API.

**Solution<a name="faq_012_solution"></a>**

Run the `export` command to view all environment variables and unset the proxy-related environment variables.

```text
unset http_proxy
unset https_proxy
```

## `host index` Not Configured<a name="faq_013"></a>

**Symptom<a name="faq_013_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
You should exec "export VC_TASK_INDEX={0|1|2...}" to configure the host index
```

**Cause Analysis<a name="faq_013_analysis"></a>**

`hosts.conf` has not been configured correctly.

**Solution<a name="faq_013_solution"></a>**

Configure `hosts.conf` correctly based on the IP addresses of the single-node or multi-node environment.

```shell
# host, index, train_master_index, infer_master_index (optional)
# If training and inference are deployed on the same node in a single-node environment,
# infer_master_index must be configured.

# [Single-node training + inference]
# Example 1: Single node, with training and inference deployed on the same node for local testing
# host, index, train_master_index, infer_master_index (optional)
# 192.168.0.1,0,1,1

# [Multi-node training + inference]
# Example 2: Two nodes, with training and inference deployed separately
192.168.0.1,0,0
192.168.0.2,1,1
```

## Ray Startup Failed<a name="faq_014"></a>

**Symptom<a name="faq_014_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
ConnectionError: Could not find any running Ray instance. Please specify the one to connect to by setting the --address flag or RAY_ADDRESS environment variable.
```

**Cause Analysis<a name="faq_014_analysis"></a>**

Ray failed to start due to an environment issue.

**Solution<a name="faq_014_solution"></a>**

Try starting Ray manually to determine whether the issue is caused by the environment. Before running the following command, ensure that the specified ports are not in use.

```shell
ray start --head --port=7894 --dashboard-port=7895
```

If Ray reports an error, troubleshoot the environment. The following are some possible areas to check.

1. Ensure that the ports used by Ray are not in use.
2. Ensure that sufficient file descriptors are available for the `raylet` process.
3. Check whether Ray processes are running in other containers on the machine.

## Ray Session Name Mismatch<a name="faq_015"></a>

**Symptom<a name="faq_015_phenomenon"></a>**

After starting the script, the following error message is displayed.

```text
AssertionError: Session name session_2026-07-09_09-17-24_537353_45519 does not match persisted value b'session_2026-07-08_16-03-21_014733_24109'. Perhaps there was an error connecting to Redis.
```

**Cause Analysis<a name="faq_015_analysis"></a>**

Agent SDK uses Ray as the underlying framework for distributed training, and Ray relies on Redis to manage cluster status. The previous Ray session did not exit properly, and its temporary files or metadata in Redis were not cleared, causing a conflict with the newly started session.

**Solution<a name="faq_015_solution"></a>**

Enter the container where the previous Ray session was not properly cleaned up and run the following commands to clean up residual Ray processes and temporary files.

```shell
ray stop -f
rm -rf /tmp/ray
```

Alternatively, stop the other containers directly.

```shell
docker stop <container_name>
```
