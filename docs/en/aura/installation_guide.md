# Installation and Deployment<a name="ZH-CN_TOPIC_0000002492554169"></a>

## Obtaining the Installation Package<a name="ZH-CN_TOPIC_0000002459514672"></a>

This section describes how to obtain the required software package and its corresponding digital signature file.

**Table 1** Software packages

|Component|Software Package|How to Obtain|
|--|--|--|
|Agent SDK|Agent package|Link (to be updated)|

**Verifying the Software Digital Signature<a name="section10830205518487"></a>**

To prevent the package from being maliciously tampered with during transmission or storage, download the digital signature file for integrity check while downloading the package.

After downloading the software package, verify its PGP digital signature according to the *OpenPGP Signature Verification Guide*. If the verification fails, do not use the software package, and contact Huawei technical support.

Before you use a software package for installation or upgrade, perform the preceding operations to verify its digital signature to ensure that the software package is not tampered with.

For carrier users, visit [https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction).

For enterprise users, visit [https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054).

**Precautions<a name="section59421949184112"></a>**

If you need to install third-party software other than the Agent SDK software package, upgrade the software to the latest version in a timely manner and fix existing vulnerabilities.

## Installation Prerequisites<a name="ZH-CN_TOPIC_0000002492554221"></a>

### Installing Ubuntu System Dependencies<a name="ZH-CN_TOPIC_0000002492554173"></a>

For the names, corresponding versions, and acquisition suggestions of dependencies required in the Ubuntu environment, see [Table 1 Ubuntu dependencies and corresponding versions](#table20540329125613).

**Table 1** Ubuntu dependencies and corresponding versions<a id="table20540329125613"></a>

|Dependency|Recommended Version|Acquisition Suggestion|
|--|--|--|
|Python|3.10 or later|Obtain the source package from the Python official website and compile and install it.|
|CMake|4.1.0 or later|Install using a package manager. The installation command is as follows:<br>sudo apt-get install -y cmake<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|
|Make|4.3 or later|Install using a package manager. The installation command is as follows:<br>sudo apt-get install -y make<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|
|GCC|11.4.0 or later|Install using a package manager. The installation command is as follows:<br>sudo apt-get install -y gcc<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|
|G++|11.4.0 or later|Install using a package manager. The installation command is as follows:<br>sudo apt-get install -y g++<br>If the version in the package manager does not meet the minimum version requirement, you can install the package using the source code.|

Run the following commands to query the version information of dependency packages such as GCC, G++, Make, CMake, and Python to confirm whether they are installed:

```text
gcc --version
g++ --version
make --version
cmake --version
python3 --version
```

If the information similar to the following is returned, the corresponding software has been installed:

```text
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
GNU Make 4.3
cmake version 4.1.0
Python 3.11.13
```

### Installing the NPU Driver, Firmware, and CANN<a name="ZH-CN_TOPIC_0000002459514664"></a>

**Downloading Dependency Software Packages<a name="section119752030133014"></a>**

**Table 1** Software package list

<table>
<tr>
<th>Software Type</th>
<th>Package</th>
<th>How to Obtain</th>
</tr>
<tr>
<td> Ascend NPU driver</td>
<td>Ascend-hdk-{npu_type}-npu-driver_{version}_linux-{arch}.run</td>
<td rowspan="5">Click <a href="https://www.hiascend.com/developer/download/commercial/result?module=cann">this link</a>. Configure the package in Edit Resource Selection under supporting resources on the left, filter the required software package, and obtain the package after you confirm the version information.</td>
</tr>
<tr>
<td> Ascend NPU firmware</td>
<td>Ascend-hdk-{npu_type}-npu-firmware_{version}.run</td>
</tr>
<tr>
<td>CANN package</td>
<td>Ascend-cann-toolkit_{version}_linux-{arch}.run</td>
</tr>
<tr>
<td>CANN operator package</td>
<td>Ascend-cann-{npu_type}-ops_{version}_linux-{arch}.run</td>
</tr>
<tr>
<td>CANN nnal package</td>
<td>Ascend-cann-nnal_{version}_linux-{arch}.run</td>
</tr>
</table>

>[!NOTE]NOTE
>
>- `{npu_type}` is the chip name.
>- `{version}` is the software version number.
>- `{arch}` is the CPU architecture.

**Installing the NPU Driver, Firmware, and CANN<a name="section2121626113418"></a>**

1. For details, see "Installing the NPU Driver and Firmware" (commercial edition) or "Installing the NPU Driver and Firmware" (community edition) in the *CANN Software Installation Guide*.
2. For details, see "Installing CANN" (commercial edition) or "Installing CANN" (community edition) in the *CANN Software Installation Guide*.

    >[!NOTE]NOTE
    >
    >- The user installing CANN (toolkit and nnal), NPU driver, firmware, and Agent SDK must be the same user, preferably a common user.
    >- When CANN is installed, dependencies of CANN also need to be installed to ensure the normal use of Agent SDK.

### Installing Open-Source Software<a name="ZH-CN_TOPIC_0000002461034756"></a>

When you use Agent SDK for mindspeed-rl training, install the following open-source software.

Run the following commands to install the specified versions of the repositories in the specified location. For a common user, use a path that the user has permission to access.

```bash
mkdir -p /home/third-party # You can use a custom directory.
cd /home/third-party

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cd ..

git clone https://github.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 2.1.0_core_r0.8.0
cd ..

git clone https://github.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 2.1.0
cd ..

git clone https://github.com/Ascend/MindSpeed-RL.git
cd MindSpeed-RL
git checkout 2.2.0
cd ..

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
VLLM_TARGET_DEVICE=empty pip3 install -e .
cd ..

pip3 install --ignore-installed --upgrade blinker==1.9.0
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.9.1-dev
pip3 install -e .
cd ..

pip3 install -r MindSpeed/requirements.txt
pip3 install -r MindSpeed-LLM/requirements.txt
pip3 install -r MindSpeed-RL/requirements.txt

# Enable environment variables and adjust the directory based on the actual installation situation
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=$PYTHONPATH:/home/third-party/Megatron-LM/:/home/third-party/MindSpeed/:/home/third-party/MindSpeed-LLM:/home/third-party/MindSpeed-RL
```

### Installing Python Dependencies<a name="ZH-CN_TOPIC_0000002492474289"></a>

To use Agent SDK functions, install the following dependencies.

**Table 1**  Dependencies and recommended versions

| Dependency                 | Recommended Version       | Acquisition Suggestion                                                            |
|-----------------------|-------------|------------------------------------------------------------------|
| transformers          | 4.52.3      | Install it using pip. The installation command is as follows:<br>pip3 install transformers==4.52.3         |
| sympy                 | 1.13.1      | Install it using pip. The installation command is as follows:<br>pip3 install sympy==1.13.1                |
| pylatexenc            | 2.10        | Install it using pip. The installation command is as follows:<br>pip3 install pylatexenc==2.10             |
| openai                | 1.99.6      | Install it using pip. The installation command is as follows:<br>pip3 install openai==1.99.6               |
| torch                 | 2.5.1       | Install it using pip. The installation command is as follows:<br>pip3 install torch==2.5.1                 |
| torch_npu             | 2.5.1.post1 | Install it using pip. The installation command is as follows:<br>pip3 install torch_npu==2.5.1.post1       |
| vertexai              | 1.64.0      | Install it using pip. The installation command is as follows:<br>pip3 install vertexai==1.64.0             |
| sentence_transformers | 5.1.0       | Install it using pip. The installation command is as follows:<br>pip3 install sentence_transformers==5.1.0 |
| hydra-core            | 1.3.2       | Install it using pip. The installation command is as follows:<br>pip3 install hydra-core==1.3.2            |
| regex                 | 2025.8.29   | Install it using pip. The installation command is as follows:<br>pip3 install regex==2025.8.29             |
| tensordict            | 0.1.2       | Install it using pip. The installation command is as follows:<br>pip3 install tensordict==0.1.2            |
| word2number           | 1.1         | Install it using pip. The installation command is as follows:<br>pip3 install word2number==1.1             |
| codetiming            | 1.4.0       | Install it using pip. The installation command is as follows:<br>pip3 install codetiming==1.4.0            |
| torchvision           | 0.20.1      | Install it using pip. The installation command is as follows:<br>pip3 install torchvision==0.20.1          |
| ray                   | 2.42.1      | Install it using pip. The installation command is as follows:<br>pip3 install ray==2.42.1                  |
| datasets              | 4.4.1       | Install it using pip. The installation command is as follows:<br>pip3 install datasets==4.4.1              |

## Installing Agent SDK<a name="ZH-CN_TOPIC_0000002459514676"></a>

**Installation Precautions<a name="section3134195618512"></a>**

Users installing and running Agent SDK must meet the following requirements:

- You are advised to install and run Agent SDK as a common user.
- The user installing Agent SDK and the user running Agent SDK must be the same user.
- The user installing CANN (toolkit and nnal), NPU driver, firmware, and Agent SDK must be the same user, preferably a common user.
- Logs related to package installation, upgrade, uninstallation, and version queries are saved to `~/log/AgentSDK/deployment.log`. Logs related to integrity verification, file extraction, and access through the `tar` command are saved to `~/log/makeself/makeself.log`. You can view the corresponding files to complete subsequent log tracing and audit.

**Installation Procedure <a name="section12327567584"></a>**

1. Log in to the installation environment. Use the same user as the user who installed the dependencies.
2. Upload the Agent SDK package to any directory in the installation environment and go to the directory.
3. Run the installation command.

    ```bash
    chmod u+x Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run
    ./Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run --install
    ```

4. Set environment variables.

    ```bash
    export PATH=$PATH:~/.local/bin/
    ```

**References<a name="section111812571483"></a>**

**Table 1** API parameters<a id="table1361972315353"></a>

|Input Parameter|Description|
|--|--|
|--help \| -h|Queries help information.|
|--info|Queries package build information.|
|--list|Queries the file list.|
|--check|Checks package integrity.|
|--quiet \| -q|Enables the quiet mode. It must be used together with the `--install` or `--upgrade` parameter.|
|--nox11|Discarded|
|--noexec|Does not run the embedded scripts.|
|--extract=\<path>|Extracts files directly to the target directory (absolute path). It is usually used with the `--noexec` option to extract files without running them.|
|--tar arg1 [arg2 ...]|Accesses the contents of the archive file using the `tar` command.|
|--install|Installs the Agent SDK software package. The current path and installation path must not contain invalid characters. Only uppercase and lowercase letters, digits, and special characters `-_./` are supported. The installation path cannot contain any file or folder named `agent`. If a symbolic link named `agent` exists, the system prompts you to exit.|
|--install-path=*\<path>*|(Optional) Customizes the root directory for installing the software package. If it is not set, the default is the directory where the command is executed. <br>You are advised to specify an absolute path when installing Agent SDK. This parameter conflicts with the `--version` input parameter. You are not advised to install Agent SDK in `/tmp`. It must be used together with the `--install` or `--upgrade` parameter. When you use it with `--upgrade`, `--install-path` indicates the installation directory of the old package, and the upgrade runs in that directory. The input path cannot contain invalid characters. Only uppercase and lowercase letters, digits, and special characters `-_./` are supported.|
|--upgrade|Upgrades the Agent SDK software package. If an installation already exists, the system prompts you to choose whether to delete the previous installation and then reinstall Agent SDK.|
|--version|Queries the version information about the Agent SDK software package. When this operation is performed, the system temporarily installs the Agent SDK run package in `/tmp` and uninstalls it after the version number is queried.|

> [!NOTE]NOTE
>The following parameters are not displayed in the `--help` output. Do not use them directly.
>
>- `--xwin`: Runs in xwin mode.
>- `--phase2`: Requires the second phase to be executed.

# Upgrading<a name="ZH-CN_TOPIC_0000002515349619"></a>

**Procedure<a name="section13156706455"></a>**

1. Refer to [Obtaining the Installation Package](#obtaining-the-installation-package) to obtain and upload the package, and go to the Agent SDK installation directory.
2. Grant the execute permission on the software package.

    ```bash
    chmod u+x Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run
    ```

3. Run the upgrade command to upgrade the Agent SDK software package. The following is an example of the upgrade command. For details about related parameters, see [Table 1 API parameters](#table1361972315353).

    ```bash
    ./Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run --upgrade
    ```

4. When the system prompts `Do you want to upgrade by removing the old installation?`, enter `Y` or `y` to agree to delete the old installation and continue the upgrade. Enter any other character to stop the upgrade and exit the program.
5. Run the following commands to query the version upgrade records:

    ```bash
    cd ~/log/AgentSDK/
    cat deployment.log
    ```

    If the following information is displayed, the upgrade is successful:

    ```bash
    7.3.0    ->    7.3.0 Upgrade Agent SDK successfully.
    ```

# Uninstallation<a name="ZH-CN_TOPIC_0000002492474253"></a>

**Procedure<a name="section12002371094"></a>**

1. Go to the Agent SDK installation directory.

    ```bash
    cd /your_install_path
    ```

2. Run the following command to uninstall:

    ```bash
    bash agent/script/uninstall.sh
    ```

    > [!NOTE]NOTE
    >This command deletes the Agent SDK package and symbolic links in the installation directory, and deletes the `agentic_rl` package and command from the Python package directory.
