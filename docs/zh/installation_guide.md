# 安装部署<a name="ZH-CN_TOPIC_0000002492554169"></a>

## 获取安装包<a name="ZH-CN_TOPIC_0000002459514672"></a>

请参考本章获取所需软件包和对应的数字签名文件。

**表 1**  软件包

|组件名称|软件包名称|获取方式|
|--|--|--|
|Agent SDK|Agent软件包|获取链接（待更新）|


**软件数字签名验证<a name="section10830205518487"></a>**

为了防止软件包在传递过程中或存储期间被恶意篡改，下载软件包时请下载对应的数字签名文件用于完整性验证。

在软件包下载之后，请参考《OpenPGP签名验证指南》，对下载的软件包进行PGP数字签名校验。如果校验失败，请勿使用该软件包并联系华为技术支持工程师解决。

使用软件包安装/升级前，也需要按照上述过程，验证软件包的数字签名，确保软件包未被篡改。

运营商客户请访问：[https://support.huawei.com/carrier/digitalSignatureAction](https://support.huawei.com/carrier/digitalSignatureAction)

企业客户请访问：[https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054](https://support.huawei.com/enterprise/zh/tool/software-digital-signature-openpgp-validation-tool-TL1000000054)

**注意事项<a name="section59421949184112"></a>**

如需安装Agent SDK软件包以外的第三方软件，请注意及时升级最新版本，关注并修补存在的漏洞。


## 安装依赖<a name="ZH-CN_TOPIC_0000002492554221"></a>

### 安装Ubuntu系统依赖<a name="ZH-CN_TOPIC_0000002492554173"></a>

Ubuntu系统环境中所需依赖名称、对应版本及获取建议请参见[表1](#table20540329125613)。

**表 1** Ubuntu系统依赖名称对应版本<a id="table20540329125613"></a>

|依赖名称|版本建议|获取建议|
|--|--|--|
|Python|3.10及以上|请从Python官网获取源码包并进行编译安装。|
|CMake|4.1.0及以上|建议通过包管理安装，安装命令参考如下。<br>sudo apt-get install -y cmake<br>若包管理中的版本不符合最低版本要求，可自行通过源码方式安装。|
|Make|4.3及以上|建议通过包管理安装，安装命令参考如下。<br>sudo apt-get install -y make<br>若包管理中的版本不符合最低版本要求，可自行通过源码方式安装。|
|GCC|11.4.0及以上|建议通过包管理安装，安装命令参考如下。<br>sudo apt-get install -y gcc<br>若包管理中的版本不符合最低版本要求，可自行通过源码方式安装。|
|G++|11.4.0及以上|建议通过包管理安装，安装命令参考如下。<br>sudo apt-get install -y g++<br>若包管理中的版本不符合最低版本要求，可自行通过源码方式安装。|


使用如下命令查询GCC、G++、Make、CMake及Python等依赖软件包的版本信息，确认依赖软件是否已安装。

```
gcc --version
g++ --version
make --version
cmake --version
python3 --version
```

若返回如下信息，则说明相应软件已安装（以下回显仅为示例，请以实际情况为准）。

```
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
g++ (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
GNU Make 4.3
cmake version 4.1.0
Python 3.11.13
```


### 安装NPU驱动固件和CANN<a name="ZH-CN_TOPIC_0000002459514664"></a>

**下载依赖软件包<a name="section119752030133014"></a>**

**表 1**  软件包清单

<table>
<tr>
<th>软件类型</th>
<th>软件包名称</th>
<th>获取方式</th>
</tr>
<tr>
<td>昇腾NPU驱动</td>
<td>Ascend-hdk-{npu_type}-npu-driver_{version}_linux-{arch}.run</td>
<td rowspan="5">单击<a href="https://www.hiascend.com/developer/download/commercial/result?module=cann">获取链接</a>，在左侧配套资源的“编辑资源选择”中进行配置，筛选配套的软件包，确认版本信息后获取所需软件包。</td>
</tr>
<td>昇腾NPU固件</td>
<td>Ascend-hdk-{npu_type}*npu-firmware_{version}.run</td>
<tr>
</tr>
<td>CANN软件包</td>
<td>Ascend-cann-toolkit_{version}_linux-{arch}.run</td>
<tr>
</tr>
<td>CANN算子包</td>
<td>Ascend-cann-{npu_type}-ops_{version}_linux-{arch}.run</td>
<tr>
</tr>
<td>CANN nnal包</td>
<td>Ascend-cann-nnal_{version}_linux-{arch}.run</td>
<tr>
</table>


>[!NOTE] 说明
>-   \{npu\_type\}表示芯片名称。
>-   \{version\}表示软件版本号。
>-   \{arch\}表示CPU架构。

**安装NPU驱动固件和CANN<a name="section2121626113418"></a>**

1.  参考《CANN 软件安装指南》中的“安装NPU驱动和固件”章节（商用版）或“安装NPU驱动和固件”章节（社区版）安装NPU驱动固件。
2.  参考《CANN 软件安装指南》的“安装CANN”章节（商用版）或《CANN 软件安装指南》的“安装CANN”章节（社区版）安装CANN。

    >[!NOTE] 说明 
    >-   安装CANN（toolkit，nnal）、NPU驱动固件和安装Agent SDK的用户需为同一用户，建议为普通用户。
    >-   安装CANN时，为确保Agent SDK正常使用，CANN的相关依赖也需要一并安装。


### 安装开源软件<a name="ZH-CN_TOPIC_0000002461034756"></a>

通过Agent SDK使用mindspeed-rl训练时，需要安装以下开源软件。

参考如下命令，安装指定版本的仓库至指定位置，普通用户需要注意使用拥有权限的路径。

```
mkdir -p /home/third-party # 可自定义目录
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

# 使能环境变量，根据实际安装的情况调整目录
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/cann/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=$PYTHONPATH:/home/third-party/Megatron-LM/:/home/third-party/MindSpeed/:/home/third-party/MindSpeed-LLM:/home/third-party/MindSpeed-RL
```


### 安装Python软件包依赖<a name="ZH-CN_TOPIC_0000002492474289"></a>

使用Agent SDK相关功能还需要安装如下依赖。

**表 1**  依赖名称对应版本

| 依赖名称                  | 版本建议        | 获取建议                                                             |
|-----------------------|-------------|------------------------------------------------------------------|
| transformers          | 4.52.3      | 建议通过pip安装，安装命令参考如下。<br>pip3 install transformers==4.52.3         |
| sympy                 | 1.13.1      | 建议通过pip安装，安装命令参考如下。<br>pip3 install sympy==1.13.1                |
| pylatexenc            | 2.10        | 建议通过pip安装，安装命令参考如下。<br>pip3 install pylatexenc==2.10             |
| openai                | 1.99.6      | 建议通过pip安装，安装命令参考如下。<br>pip3 install openai==1.99.6               |
| torch                 | 2.5.1       | 建议通过pip安装，安装命令参考如下。<br>pip3 install torch==2.5.1                 |
| torch_npu             | 2.5.1.post1 | 建议通过pip安装，安装命令参考如下。<br>pip3 install torch_npu==2.5.1.post1       |
| vertexai              | 1.64.0      | 建议通过pip安装，安装命令参考如下。<br>pip3 install vertexai==1.64.0             |
| sentence_transformers | 5.1.0       | 建议通过pip安装，安装命令参考如下。<br>pip3 install sentence_transformers==5.1.0 |
| hydra-core            | 1.3.2       | 建议通过pip安装，安装命令参考如下。<br>pip3 install hydra-core==1.3.2            |
| regex                 | 2025.8.29   | 建议通过pip安装，安装命令参考如下。<br>pip3 install regex==2025.8.29             |
| tensordict            | 0.1.2       | 建议通过pip安装，安装命令参考如下。<br>pip3 install tensordict==0.1.2            |
| word2number           | 1.1         | 建议通过pip安装，安装命令参考如下。<br>pip3 install word2number==1.1             |
| codetiming            | 1.4.0       | 建议通过pip安装，安装命令参考如下。<br>pip3 install codetiming==1.4.0            |
| torchvision           | 0.20.1      | 建议通过pip安装，安装命令参考如下。<br>pip3 install torchvision==0.20.1          |
| ray                   | 2.42.1      | 建议通过pip安装，安装命令参考如下。<br>pip3 install ray==2.42.1                  |
| datasets              | 4.4.1       | 建议通过pip安装，安装命令参考如下。<br>pip3 install datasets==4.4.1              |




## 安装Agent SDK<a name="ZH-CN_TOPIC_0000002459514676"></a>

**安装须知<a name="section3134195618512"></a>**

安装和运行Agent SDK的用户，需要满足以下要求：

-   安装和运行Agent SDK的用户建议为普通用户。
-   安装和运行Agent SDK的用户需为同一用户。
-   安装CANN（toolkit, nnal）、NPU驱动固件和安装Agent SDK的用户需为同一用户，建议为普通用户。
-   软件包的安装、升级、卸载及版本查询相关的日志会保存至“\~/log/AgentSDK/deployment.log”文件；完整性校验、提取文件、tar命令访问相关的日志会保存至“\~/log/makeself/makeself.log”文件。用户可查看相应文件，完成后续的日志跟踪及审计。

**安装步骤<a name="section12327567584"></a>**

1.  用户登录安装环境。该用户需与安装依赖的用户为同一个用户。
2.  将Agent SDK软件包上传到安装环境的任意路径下并进入软件包所在路径。
3.  执行安装命令。

    ```
    chmod u+x Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run
    ./Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run --install
    ```

4.  设置环境变量。

    ```
    export PATH=$PATH:~/.local/bin/
    ```

**相关参考<a name="section111812571483"></a>**

**表 1**  接口参数表<a id="table1361972315353"></a>

|输入参数|含义|
|--|--|
|--help \| -h|查询帮助信息。|
|--info|查询包构建信息。|
|--list|查询文件列表。|
|--check|查询包完整性。|
|--quiet \| -q|启用静默模式。需要和--install或--upgrade参数配合使用。|
|--nox11|废弃接口，无实际作用。|
|--noexec|不运行嵌入的脚本。|
|--extract=\<path>|直接提取到目标目录（只支持绝对路径）。通常与--noexec选项一起使用，仅用于提取文件而不运行它们。|
|--tar arg1 [arg2 ...]|通过tar命令访问归档文件的内容。|
|--install|AgentSDK软件包安装操作命令。当前路径和安装路径不能存在非法字符，仅支持大小写字母、数字、-_./特殊字符。安装路径下不能存在名为agent的文件或文件夹。若存在名为agent的软链接，则会提示退出。|
|--install-path=*\<path>*|（可选）自定义软件包安装根目录。如未设置，默认为当前命令执行所在目录。建议用户使用绝对路径安装AgentSDK，指定安装路径时请避免使用相对路径。与--version输入参数有冲突，不建议在/tmp路径下安装AgentSDK。需要和--install或--upgrade参数配合使用。与--upgrade参数配合使用时，--install-path代表旧软件包的安装目录，并在该目录下执行升级。传入的路径参数不能存在非法字符，仅支持大小写字母、数字、-_./特殊字符。|
|--upgrade|Agent SDK软件包升级操作命令。如果已存在安装，将提示用户是否需要删除历史安装，之后重新安装Agent SDK。|
|--version|查询Agent SDK软件包的版本信息。执行此操作时，会在/tmp下临时安装Agent SDK的run包，查询完版本号后再卸载。|


> [!NOTE] 说明 
>以下参数未展示在--help参数中，用户请勿直接使用。
>-   --xwin：使用xwin模式运行。
>-   --phase2：要求执行第二步动作。


# 升级<a name="ZH-CN_TOPIC_0000002515349619"></a>

**操作步骤<a name="section13156706455"></a>**

1.  请参见[获取安装包](#获取安装包)获取并上传软件包，并已进入Agent SDK的安装目录。
2.  增加对软件包的可执行权限。

    ```
    chmod u+x Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run
    ```

3.  使用软件包升级命令升级当前AgentSDK软件包，升级命令如下，相关参数说明请参见[表1 接口参数表](#table1361972315353)。

    ```
    ./Ascend-mindsdk-agentsdk_7.3.0_linux-aarch64.run --upgrade
    ```
4.  升级过程中提示Do you want to upgrade by removing the old installation?时，输入Y或者y，表示同意删除旧的安装，继续进行升级；输入其他字符时停止升级，退出程序。
5.  执行如下命令可查询版本升级记录。

    ```
    cd ~/log/AgentSDK/
    cat deployment.log
    ```

    升级成功后，显示如下：

    ```
    7.3.0    ->    7.3.0 Upgrade Agent SDK successfully.
    ```


# 卸载<a name="ZH-CN_TOPIC_0000002492474253"></a>

**操作步骤<a name="section12002371094"></a>**

1.  进入Agent SDK的安装目录。

    ```
    cd /your_install_path
    ```

2.  执行如下命令，开始执行卸载。

    ```
    bash agent/script/uninstall.sh
    ```

    > [!NOTE] 说明 
    >使用该命令会删除安装目录下的Agent SDK包和软链接，同时会删除Python包目录下的agentic\_rl包和命令。


