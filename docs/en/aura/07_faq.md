# FAQ<a name="faq_main"></a>

## Network Unavailable<a name="faq_001"></a>

**Symptom<a name="faq_001_phenomenon"></a>**

After you start Aura, the following error appears.

```text
eth0: error fetching interface information: Device not found
[ERROR] [ 26-05-21 10:47:56 ] can not get IP from eth0
```

**Cause Analysis<a name="faq_001_analysis"></a>**

By default, the system obtains the local IP address through the environment variable `DEFAULT_SOCKET_IFNAME` (the default value is `eth0`). This error occurs because the network interface named `eth0` cannot be found in `ifconfig`, and therefore the correct local IP address cannot be resolved.

**Solution<a name="faq_001_solution"></a>**

Check the network configuration in your container, and change the value of the environment variable `DEFAULT_SOCKET_IFNAME` to the name of a network interface that actually exists in the current environment and contains the correct local IP address (for example, `br0`, `eth0`, or `enp189s0f0`).

The following is a specific example:

1. Run the `ifconfig` command to view the network configuration:

    ```shell
    ifconfig
    ```

2. Assume that the printed information (partial) is as follows:

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. Assume that the local IP address is 192.168.0.1. In this case, the value that points to the virtual bridge corresponding to the local IP address is `enp189s0f0`. You are advised to write the configuration to `aura/configs/env/env.local` so that it remains effective on subsequent startups:

    ```shell
    echo 'DEFAULT_SOCKET_IFNAME=enp189s0f0' >> aura/configs/env/env.local
    ```

    If you only need temporary debugging, you can also run `export DEFAULT_SOCKET_IFNAME=enp189s0f0` in the current shell. External environment variables take priority over `env.local` and `env.conf`.

## Line Break Not Recognized<a name="faq_002"></a>

**Symptom<a name="faq_002_phenomenon"></a>**

After you start Aura, the following error appears:

```text
$'\r': command not found
```

**Cause Analysis<a name="faq_002_analysis"></a>**

Line endings in shell scripts may differ between operating systems, which causes parsing errors.

**Solution<a name="faq_002_solution"></a>**

Convert all shell scripts and configuration files to the Unix format.

```shell
# Convert all shell scripts to the Unix format
find /path/to/AgentSDK -type f -name "*.sh" -exec dos2unix {} +
```

## IP Address or Port Already Bound<a name="faq_003"></a>

**Symptom<a name="faq_003_phenomenon"></a>**

After you start Aura, the following error appears:

```text
RuntimeError: createHCCLCommOrigin:build/CMakeFiles/torch_npu.dir/compiler_depend.ts:2314 HCCL function error: HcclGetRootInfo(&hcclID), error code is 7
ERR02200 DIST call hccl api failed
Failed to bind the IP port. Reason: The IP address and port have been bound already.
```

**Cause Analysis<a name="faq_003_analysis"></a>**

Another process occupies the HCCL port. Therefore, HCCL reports an error indicating that the IP address and port are already bound.

**Solution<a name="faq_003_solution"></a>**

Set environment variables to change the HCCL port:

```shell
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60100
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
```

## Inconsistent `uid`<a name="faq_004"></a>

**Symptom<a name="faq_004_phenomenon"></a>**

In co-located mode, after training and inference are complete, the following error appears in the last verification step:

```text
AssertionError: 'uid' in tensor_dict1 and tensor_dict2 are not same object..
```

**Cause Analysis<a name="faq_004_analysis"></a>**

The `test_freq` parameter is set incorrectly in the YAML file.

**Solution<a name="faq_004_solution"></a>**

Set the `test_freq` parameter in the YAML file to -1 to disable the verification step.

## Some Fields in the `utils` File Not Found When Running Triton<a name="faq_005"></a>

**Symptom<a name="faq_005_phenomenon"></a>**

After you start the script, `triton/backends/ascend/utils.py` reports that `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE` cannot be found:

```text
/tmp/tmpaxgs3vq5/npu_utils.cpp:324:3: error: could not convert '{{"LOW_POWER_TIMEOUT", RT_LIMIT_TYPE_LOW_POWER_TIMEOUT}, {"WARP_STACK_SIZE", <expression error>}, {"DVG_WARP_STACK_SIZE", RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE}, {"STACK_SIZE", RT_LIMIT_TYPE_STACK_SIZE}}' from '<brace-enclosed initializer list>' to 'const std::unordered_map<std::__cxx11::basic_string<char>, tagRtLimitType>' [repeated 3x across cluster]
```

**Cause Analysis<a name="faq_005_analysis"></a>**

CANN 8.5.1 contains this file, whereas CANN 9.0.0 has removed it. You can use the following command to search for the location of this file.

```text
grep -R RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE /usr/local/Ascend/ascend-toolkit/latest/include/experiment/runtime/runtime
```

triton-ascend 3.2.0 has this issue. You can apply the following patches to fix it:

- [triton-ascend_driver.patch](../../../aura/third_party/patch/triton-ascend_driver.patch)
- [triton-ascend_npu_utils.patch](../../../aura/third_party/patch/triton-ascend_npu_utils.patch)

The corresponding PR for triton-ascend: <https://atomgit.com/Ascend/triton-ascend/pull/1525/diffs>

**Solution<a name="faq_005_solution"></a>**

You can directly run the provided script to apply the patches with one click:

```bash
bash /home/work/AgentSDK/docker/aura/patch/patch_triton_ascend.sh
```

You can also manually patch this part of triton-ascend 3.2.0.

## Missing Module `pkg_resources`<a name="faq_006"></a>

**Symptom<a name="faq_006_phenomenon"></a>**

After you start the script, the following error appears:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

**Cause Analysis<a name="faq_006_analysis"></a>**

The `setuptools` version is relatively new, and this module has been removed in the new version.

**Solution<a name="faq_006_solution"></a>**

Install `setuptools` 80.10.2.

## Extra `strict` Field in the Configuration File<a name="faq_007"></a>

**Symptom<a name="faq_007_phenomenon"></a>**

After you start the script, the following error appears:

```text
TypeError("CheckpointConfig.__init__() got an unexpected keyword argument 'strict'")
```

**Cause Analysis<a name="faq_007_analysis"></a>**

The configuration file reads verl in the root directory. The version of verl in the root directory is incorrect. You are advised to switch to the specified commit version.

**Solution<a name="faq_007_solution"></a>**

Switch the version of verl in the root directory to the specified commit version.

```text
cd /
git clone https://github.com/verl-project/verl.git
cd verl
git checkout e9972368aa6a6078eacd7f0678bdfdd0196ce7b5
```

## Missing `fused_moe.runner` Module in vllm<a name="faq_008"></a>

**Symptom<a name="faq_008_phenomenon"></a>**

After you start the script, the following error appears:

```text
ModuleNotFoundError: No module named 'vllm.model_executor.layers.fused_moe.runner'
```

**Cause Analysis<a name="faq_008_analysis"></a>**

vllm 0.16.0rc1 does not have this runner directory. It is available only from 0.16.1rc0 onward. You are advised to switch directly to the specified commit version.

For the version mapping between vllm and vllm-ascend, see [vllm-ascend version release policy](https://docs.vllm.com.cn/projects/ascend/en/latest/community/versioning_policy.html).

**Solution<a name="faq_008_solution"></a>**

Switch vllm to the corresponding version.

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

## Missing `libtorch_cuda.so` File<a name="faq_009"></a>

**Symptom<a name="faq_009_phenomenon"></a>**

After you start the script, the following error appears:

```text
OSError: libtorch_cuda.so: cannot open shared object file: No such file or directory.
```

**Cause Analysis<a name="faq_009_analysis"></a>**

You are calling the CUDA version of torch. On Ascend platforms, call either TorchNPU or the CPU version of torch. Therefore, uninstall torch and install the CPU version of torch.

**Solution<a name="faq_009_solution"></a>**

Install the CPU version of torch.

```text
pip uninstall -y torch torchvision torchaudio
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
```

## Missing `language.target_info` Module in triton<a name="faq_010"></a>

**Symptom<a name="faq_010_phenomenon"></a>**

After you start the script, the following error appears:

```text
Failed to import Triton kernels. Please make sure your triton version is compatible. Error: No module named 'triton.language.target_info'
```

**Cause Analysis<a name="faq_010_analysis"></a>**

Here, vllm checks whether the current environment is a CUDA environment. If it is not a CUDA environment, vllm throws an ERROR and then enters the ascend-vllm process.

**Solution<a name="faq_010_solution"></a>**

This is internal vllm code logic and requires no fix. It only prints a prompt and does not interrupt the main process. You are advised to troubleshoot other error causes.

## Missing `moe_distribute_comm_ctx.h` File When Compiling and Installing vllm-ascend<a name="faq_011"></a>

**Symptom<a name="faq_011_phenomenon"></a>**

After you start the script, the following error appears:

```text
fatal error: 'moe_distribute_comm_ctx.h' file not found
```

**Cause Analysis<a name="faq_011_analysis"></a>**

For an A3 server, when vllm-ascend compiles operators with cmake, the header file path is missing. Add an environment variable so that the file path is added to the file paths in CMakeLists.

**Solution<a name="faq_011_solution"></a>**

Add an environment variable to add the header file path included when cmake compiles operators:

```text
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe/impl/ops_transformer/ascendc/common/inc/kernel:${CPLUS_INCLUDE_PATH}
```

## HTTP Connection Failure<a name="faq_012"></a>

**Symptom<a name="faq_012_phenomenon"></a>**

After you start the script, the following error appears:

```text
httpcore.ConnectError: All connection attempts failed
```

**Cause Analysis<a name="faq_012_analysis"></a>**

An HTTP proxy exists in the environment, which causes the network connection to fail when InferExecutor calls a vLLM API.

**Solution<a name="faq_012_solution"></a>**

Use the `export` command to view all environment variables, and unset the proxy-related environment variables.

```text
unset http_proxy
unset https_proxy
```

## Host Index Not Configured<a name="faq_013"></a>

**Symptom<a name="faq_013_phenomenon"></a>**

After you start the script, the following error appears:

```text
You should exec "export VC_TASK_INDEX={0|1|2...}" to configure the host index
```

**Cause Analysis<a name="faq_013_analysis"></a>**

You have not configured `hosts.conf` correctly.

**Solution<a name="faq_013_solution"></a>**

Configure `hosts.conf` correctly according to the IP address for single-server or multi-server deployment:

```shell
# host,index,train_master_index,infer_master_index(optional)
# If training and inference are co-deployed on a single server, you need to configure infer_master_index

# [Single-Server Training and Inference]
# Configuration example 1: single-server, co-located training and inference deployment, convenient for local debugging
# host,index,train_master_index,infer_master_index(optional)
# 192.168.0.1,0,1,1

# [Multi-Server Training and Inference]
# Configuration example 2: two servers, disaggregated training and inference, separate node deployment
# host,index,train_master_index,infer_master_index(optional)
192.168.0.1,0,0
192.168.0.2,1,1
```

## Ray Startup Failure<a name="faq_014"></a>

**Symptom<a name="faq_014_phenomenon"></a>**

After you start the script, the following error appears:

```text
ConnectionError: Could not find any running Ray instance. Please specify the one to connect to by setting the --address flag or RAY_ADDRESS environment variable.
```

**Cause Analysis<a name="faq_014_analysis"></a>**

Environment issues cause Ray to fail to start.

**Solution<a name="faq_014_solution"></a>**

Try to start Ray manually to determine whether the environment has an issue. Before you run the following command, ensure that the port used by the command is not occupied:

```shell
ray start --head --port=7894 --dashboard-port=7895
```

If Ray reports an error, troubleshoot the environment. The following are some troubleshooting directions:

1. Ensure that the port used by Ray is not occupied.
2. Ensure that there are enough file descriptors for the raylet process.
3. Check whether Ray processes are running in other containers on the machine.

## Ray Session Name Mismatch<a name="faq_015"></a>

**Symptom<a name="faq_015_phenomenon"></a>**

After you start the script, the following error appears:

```text
AssertionError: Session name session_2026-07-09_09-17-24_537353_45519 does not match persisted value b'session_2026-07-08_16-03-21_014733_24109'. Perhaps there was an error connecting to Redis.
```

**Cause Analysis<a name="faq_015_analysis"></a>**

Agent SDK uses Ray under the hood for distributed training, and Ray depends on Redis to manage the cluster state. The previous Ray session did not exit properly. Therefore, its temporary files or the metadata in Redis were not cleaned up, which conflicts with the newly started session.

**Solution<a name="faq_015_solution"></a>**

Enter the container in which Ray was not cleaned up previously, and run the following commands to remove the residual Ray processes and temporary files:

```shell
ray stop -f
rm -rf /tmp/ray
```

Or stop the other containers directly:

```shell
docker stop <container name>
```
