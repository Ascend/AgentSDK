# 为Ascend Agent SDK 贡献

感谢您考虑为Ascend Agent SDK 做出贡献！我们欢迎任何形式的贡献，包括缺陷修复、功能增强、测试补充、文档改进以及使用反馈。无论您是第一次参与开源项目，还是已经具备丰富经验，您的贡献都非常宝贵。

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

1. **Fork 仓库到个人账号**

   在 GitCode 上将官方仓库 Fork 到个人空间。

2. **克隆个人仓库到本地**

   ```bash
   git clone https://gitcode.com/<your-username>/AgentSDK.git
   cd AgentSDK
   ```

3. **创建开发分支**

   ```bash
   git checkout -b feature/<your-feature-name>
   # 或
   git checkout -b fix/<issue-id>
   ```

4. **进行代码开发**

   尽量保证改动聚焦、可审查、可回滚，并补充相应测试。

5. **执行本地测试**

   提交前请至少完成与改动相关的本地验证，确保本地相关测试通过。

6. **执行 pre-commit 检查**

   本地提交代码前请先执行 pre-commit 检查，确保代码风格与安全检查通过。

7. **提交 Pull Request**

   - 保持 PR 小规模：一次 PR 只解决一个问题，建议单个 PR 的代码变更量控制在 1000 行以内（含测试）
   - 及时更新：定期同步上游主分支，及时响应评审意见
   - 清晰描述：详细描述变更原因和方案，提供测试方法，如有必要添加截图、示例或对比结果

8. **社区评审与合入**

   PR 需满足项目评审要求，至少获得 2 位 Maintainer 或 Committer 的 `/lgtm` 以及 1 个 `/approve` 后，由 Maintainer 或 Committer 合入；禁止合并自己的 PR。

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

- 开发规范
  - [《Ascend Python 编码风格指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-coding-style-guide.md)
- 安全编程指导
  - [《Ascend Python 安全编程指南》](https://gitcode.com/Ascend/community/blob/master/docs/contributor/Ascend-python-secure-coding-guide.md)
- 更多社区相关规范，请访问[Ascend社区community](https://gitcode.com/Ascend/community)
