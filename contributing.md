# 为Ascend Agent SDK贡献

感谢您考虑为Ascend Agent SDK做出贡献！我们欢迎任何形式的贡献，包括缺陷修复、功能增强、测试补充、文档改进以及使用反馈。无论您是第一次参与开源项目，还是已经具备丰富经验，您的贡献都非常宝贵。

您可以通过以下方式参与 Agent SDK 社区建设：

- 通过[Agent SDK新手任务池](https://gitcode.com/Ascend/AgentSDK/issues/59)参与贡献
- 通过 [Issues](https://gitcode.com/Ascend/AgentSDK/issues) 反馈缺陷、提出建议或讨论需求
- 提交代码，修复问题或实现新功能
- 为已有功能补充测试用例，提升稳定性和可维护性
- 改进用户文档、接口文档和示例内容
- 参与 Pull Request 评审，帮助其他贡献者完善实现
- 传播项目：在博客文章、社交媒体上分享Agent SDK，或给仓库点个⭐。

参与贡献前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)，完成签署后才能提交 Pull Request，未签署的 PR 将无法合入。在开始之前，请先阅读 [Agent SDK 项目说明](./README.md)。

## 贡献方式

### Pull Request

提交 PR 前，请先了解[PR最佳实践](#pr最佳实践)，掌握从 Fork 到提交、从代码审查到合并的完整 PR 流程，包括 PR 评审要求与合并规范。

### Issue

通过 [Issues](https://gitcode.com/Ascend/AgentSDK/issues) 反馈缺陷、提出建议或讨论需求，我们会尽快回复。

### SIG会议

Agent SDK 社区通过 SIG 例会进行技术交流与议题评审，可提前在[会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-AgentSDK)中查看例会安排，SIG 信息与成员列表参见 [SIG 成员列表](https://meeting.ascend.osinfra.cn/?sig=sig-AgentSDK)。

## PR最佳实践

1. **Fork仓库**

   在GitCode平台代码仓库右上角点击"Fork"按钮，Fork一份源代码到个人仓。

2. **克隆到本地**

   将Fork到个人仓的代码克隆到本地进行代码开发。

   ```bash
   git clone https://gitcode.com/<your-username>/AgentSDK.git
   cd AgentSDK
   ```

3. **创建开发分支**

   ```bash
   git checkout -b {new_branch_name} origin/master
   ```

4. **代码开发**

   质量符合[开发规范](#dev-rule)和[安全编程指导](#sec-guide)。

5. **运行测试**

    1. 拉取CI流水线镜像环境

       该镜像已具备构建验证的所有基础环境，开发者无需安装任何额外模块，也无需执行 `pip install -e .`。

       ```bash
       docker pull swr.cn-north-4.myhuaweicloud.com/ascend-mindx/mindx_arm:SDK_20260112_1
       ```

    2. 下载项目运行所需的第三方仓库

       CI 镜像仅提供基础环境，项目依赖的第三方源码仓库需通过脚本单独下载。

       ```bash
       cd AgentSDK/aura
       bash download_third_party.sh
       ```

    3. 在提交代码前，请补充测试用例并确保所有测试通过，本地执行UT。

       单元测试按模块拆分为两个独立脚本，CI 流水线可按需调用：

       ```bash
       cd AgentSDK
       # aura 单元测试
       bash script/aura_ut.sh

       # openclaw 单元测试
       bash script/openclaw_ut.sh
       ```

       预冒烟测试同样按模块拆分：

       ```bash
       cd AgentSDK
       # aura 预冒烟
       bash run_presmoke_aura.sh

       # openclaw 预冒烟
       bash run_presmoke_openclaw.sh
       ```

6. **执行 pre-commit 检查**

   本地提交代码前请先执行pre-commit检查，检查指导参见[pre-commit本地运行指南](https://gitcode.com/Ascend/community/blob/master/docs/contributor/pre-commit-guide.md)。

7. **提交 Pull Request**

    - 保持 PR 小规模：一次 PR 只解决一个问题，建议单个 PR 的代码变更量控制在 1000 行以内（含测试）。
    - 及时更新：定期同步上游主分支，及时响应评审意见。
    - 清晰描述：详细描述变更原因和方案，提供测试方法，如有必要添加截图、示例或对比结果。

8. **社区评审与合入**

    - PR 需满足项目评审要求，至少获得 2 位 Maintainer 或 Committer 的 `/lgtm` 以及 1 个 `/approve` 后，由 Maintainer 或 Committer 合入；禁止合并自己的 PR。
    - 如果涉及patch、头文件宏、API接口等更新，需提交社区在SIG例会进行评审，社区定期例会与活动参见[会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-AgentSDK)。

## 分支/Tag命名规则

### 自研代码仓库

| 分支类型 | 分支名规则 | 示例 | 说明 | tag名规则 | tag示例 |
|---------|---------|------|------|------|------|
| 主干&开发 | `master` | `-` | `-` |`-` |`-` |
| release | `release/<版本号>` | `release/v26.1.0` | 正式版本 |`<版本号>[-beta.<序号>]` |`v26.1.0` ， `v26.1.0-beta.1`|
| 预研 | `spike/<基线分支>/<描述>` | `spike/release-v26.1.0/auth-redesign` | 不合入主干，后续删除 |`-` |`-` |
| poc | `poc/<基线分支>/<描述>` | `poc/release-v26.1.0/auth-redesign` | 后续合入主干 |`poc/<基线分支>/<描述>-v<序号>` |`poc/release-v26.1.0/auth-redesign-v1`|
| 临时 | `tmp/<描述>` | `tmp/pre-commit` | 不合入主干，后续删除 |`-` |`-` |

### Fork开源社区代码仓库

| 分支类型 | 分支名规则 | 示例 | 说明 | tag名规则 | tag示例 |
|---------|---------|------|------|------|------|
| 社区分支 | `-` | `v2.1.0` | 不合入代码 |`-` |`-` |
| release | `release/<社区分支>-<产品版本号>` | `release/v2.1.0-26.0.0` | 正式版本开发分支 |`v<产品版本号>-<社区分支>` |`v26.0.0-2.1.0`|
| 预研 | `spike/<基线分支>/<描述>` | `spike/release-v26.1.0/auth-redesign` | 不合入release分支，后续删除 |`-` |`-` |
| poc | `poc/<基线分支>/<描述>` | `poc/release-v26.1.0/auth-redesign` | 后续合入release分支 |`poc/<基线分支>/<描述>-v<序号>` |`poc/release-v26.1.0/auth-redesign-v1`|
| 临时 | `tmp/<描述>` | `tmp/pre-commit` | 不合入release分支，后续删除 |`-` |`-` |

## 参考

- 开发规范<a id="dev-rule"></a>
    - [《Ascend Python 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- 安全编程指导<a id="sec-guide"></a>
    - [《Ascend Python 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-secure-coding-guide.md)
- 更多社区相关规范，请访问[Ascend社区community](https://gitcode.com/Ascend/community)
