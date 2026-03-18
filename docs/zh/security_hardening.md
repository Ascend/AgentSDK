
# 安全加固<a name="ZH-CN_TOPIC_0000002459514612"></a>

## 安全要求<a name="ZH-CN_TOPIC_0000002459355012"></a>

使用命令行接口读取文件时，用户需要保证该文件的owner必须为自己，且权限不大于640，避免发生提权等安全问题。

外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。


## 加固需知<a name="ZH-CN_TOPIC_0000002459514624"></a>

本文中列出的安全加固措施为基本的加固建议项。用户应根据自身业务，重新审视整个系统的网络安全加固措施，必要时可参考业界优秀加固方案和安全专家的建议。


## 操作系统安全加固<a name="ZH-CN_TOPIC_0000002492554145"></a>

### 防火墙配置<a name="ZH-CN_TOPIC_0000002459514636"></a>

操作系统安装后，若配置普通用户，可以通过在“/etc/login.defs”文件中新增“ALWAYS\_SET\_PATH=yes”配置，防止越权操作。


### Ray临时目录安全配置<a name="ZH-CN_TOPIC_0000002507137787"></a>

为确保安全，Ray的临时目录放置在运行用户的home目录下，具体路径为“\~/.ray/tmp”。同时，需要禁止Ray自动将临时目录权限设为共享。

**配置步骤<a name="section86341587297"></a>**

1.  创建补丁文件。

    创建ray\_noshare\_patch.py文件，具体如下：

    ```
    import importlib
    try:
        utils = importlib.import_module("ray._private.utils")
        def _no_share(*args, **kwargs):
            return False
        utils.try_make_directory_shared = _no_share
    except Exception:
        pass
    ```

    创建ray\_noshare.pth文件，具体如下：

    ```
    import ray_noshare_patch
    ```

2.  将补丁文件拷贝到Python的site\_packages（如“/home/HwHiAiUser/.local/lib/python3.11/site-packages”）下。

> [!NOTICE] 须知 
>当前临时目录缺少自动清理机制，若Agent长时间运行，可能导致临时目录持续增长并最终占满磁盘。为避免该风险，需要用户自行清理“\~/.ray/tmp”目录。


### 设置umask<a name="ZH-CN_TOPIC_0000002492474261"></a>

建议用户将宿主机和容器中的umask设置为0027，提高文件权限。

以设置umask为0027为例，具体操作如下所示。

1.  以root用户登录服务器，编辑“/etc/profile”文件。

    ```
    vim /etc/profile
    ```

2.  在“/etc/profile”文件末尾加上**umask 0027**，保存并退出。
3.  执行如下命令使配置生效。

    ```
    source /etc/profile
    ```


### 无属主文件安全加固<a name="ZH-CN_TOPIC_0000002459355016"></a>

因为Docker镜像与物理机上的操作系统存在差异，系统中的用户可能不能一一对应，导致物理机或容器运行过程中产生的文件变成无属主文件。

用户可以执行**find / -nouser -o -nogroup**命令，查找容器内或物理机上的无属主文件。根据文件的uid和gid创建相应的用户和用户组，或者修改已有用户的uid、用户组的gid来适配，赋予文件属主，避免无属主文件给系统带来安全隐患。



### 模型保存路径安全加固<a name="ZH-CN_TOPIC_0000002459355017"></a>

训练过程中checkpoint默认保存在当前目录下的`checkpoints/${project_name}/${experiment_name}`路径中，该目录包含模型权重、优化器状态等敏感信息。为确保安全，需要对该路径进行以下安全加固：

1.  确保checkpoint目录的权限设置正确：
    -   目录权限设置为750。
    -   文件权限设置为640。

2.  目录属主应为当前用户，避免其他用户访问。

3.  避免使用软链接，防止路径遍历攻击。

4.  定期检查和清理不需要的checkpoint文件，避免敏感信息泄露。

5.  如需将checkpoint存储到其他路径，请确保目标路径同样满足上述安全要求。

可以使用以下命令检查和设置checkpoint目录的权限：

```
# 创建checkpoint目录并设置权限
mkdir -p checkpoints/${project_name}/${experiment_name}
chmod 750 checkpoints/${project_name}/${experiment_name}
chmod 750 checkpoints/${project_name}
chmod 750 checkpoints

# 对已有的checkpoint文件设置权限
find checkpoints -type d -exec chmod 750 {} \;
find checkpoints -type f -exec chmod 640 {} \;

# 验证目录属主
ls -la checkpoints/
```


## 查看命令行操作记录<a name="ZH-CN_TOPIC_0000002479124428"></a>

命令行操作日志记录在系统history中。

**查看安装、卸载的历史记录<a name="section1220492120526"></a>**

当注销系统或者退出容器时会将history中的历史命令记录保存到“\~/.bash\_history”文件中。所以，可以直接查看.bash\_history文件就能找到命令行的记录。

命令历史会先缓存在内存中，只有当终端正常退出时才会写入“\~/.bash\_history”文件。执行以下命令可立即将内存中的历史记录写入.bash\_history文件：

```
history -a
```

**修改历史记录的保存数量<a name="section56389529527"></a>**

在Linux系统中，history命令一般默认保存最新的1000条命令。如果需要修改保存的命令数量，比如只保留200条历史命令，则可以在“/etc/profile”文件中修改HISTSIZE环境变量。修改方法如下：

-   使用编辑器（如vim编辑器）修改。
-   使用sed直接修改，命令如下：

    **sed -i 's/^HISTSIZE=**_number_**/HISTSIZE=**_newNumber_**/' /etc/profile**，_number_表示修改前的命令数量，_newNumber_表示修改后的命令数量。以保存的命令数量从1000改为200为例：

    ```
    sed -i 's/^HISTSIZE=1000/HISTSIZE=200/' /etc/profile
    ```

修改完成之后需要执行**source /etc/profile**使环境变量生效。

**修改历史命令文件时间戳<a name="section18178420544"></a>**

如果需要在历史命令文件中有时间戳记录，可以在“/etc/profile”中添加如下配置：

**HISTTIMEFORMAT='%F %T '**

添加完成之后需要执行**source /etc/profile**命令使环境变量生效。添加时间戳之后，history命令结果如图所示：

```
2025-11-08 10:47:08 agentic_rl --config-path=/home/config/agentic_parameters.yaml
2025-11-08 10:47:08 agentic_rl --config-path=/home/config/agentic_parameters.yaml
2025-11-08 14:25:58 histroy | grep "agentic_rl"
2025-11-08 14:26:03 history | grep "agentic_rl"
```

此外，如果需要将历史命令记录在自定义文件中，可以在“/etc/profile”中设置HISTFILE环境变量，设置完成之后执行**source /etc/profile**命令使环境变量生效。比如：

```
HISTDIR=~/log/AgentSDK   # 配置历史命令记录保存文件
HISTFILE="$HISTDIR/AgentSDK.log"
mkdir -p $HISTDIR
chmod 750 $HISTDIR
touch $HISTFILE
chmod 640 $HISTFILE
USER_IP=`who -u am i 2>/dev/null| awk '{print $NF}'|sed -e 's/[()]//g'`
if [ -z $USER_IP ]
then
  USER_IP=`hostname`
fi
export HISTTIMEFORMAT="%F %T $USER_IP:`whoami` "    # history命令显示格式：时间、IP、用户名、执行命令 
PROMPT_COMMAND=' { date "+%Y-%m-%d %T - $(history 1 | { read x cmd; echo "$cmd"; })"; } >> $HISTFILE'    # 实时将history命令写到配置的文件里
```

其中日志文件路径为“\~/log/AgentSDK”，请保证磁盘空间足够，日志文件设置权限为640。

