[TOC]

# 为 AgentSDK 贡献

感谢您考虑为 AgentSDK 做出贡献！我们欢迎任何形式的贡献，包括缺陷修复、功能增强、测试补充、文档改进以及使用反馈。无论您是第一次参与开源项目，还是已经具备丰富经验，您的贡献都非常宝贵。

在开始之前，请先阅读以下文档：

* [AgentSDK 项目说明](README.md)

## 贡献方式

您可以通过以下方式参与 AgentSDK 社区建设：

- 通过 [Issues](https://gitcode.com/Ascend/AgentSDK/issues) 反馈缺陷、提出建议或讨论需求
- 提交代码，修复问题或实现新功能
- 为已有功能补充测试用例，提升稳定性和可维护性
- 改进用户文档、接口文档和示例内容
- 参与 Pull Request 评审，帮助其他贡献者完善实现

## 贡献流程

### 开发与测试

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

   开发过程中请遵循本文档中的[代码规范](#代码规范)，并尽量保证改动聚焦、可审查、可回滚。

5. **执行本地测试**

   提交前请至少完成与改动相关的本地验证，具体请参见[代码测试](#代码测试)。

6. **更新相关文档**

   如果您的修改影响用户使用方式、配置方法、接口行为或输出结果，请同步更新文档，具体请参见[文档开发](#文档开发)。

7. **提交 Pull Request**

   完成开发后，请参见[Pull Request](#pullrequest)章节提交代码并参与评审流程。

# 代码规范

## Python 代码规范

- 遵循 PEP 8 编码规范
- 使用 4 个空格进行缩进
- 类名使用大驼峰命名法，例如 `DataManager`
- 函数、变量和模块使用小写加下划线命名法，例如 `parse_config`
- 新增或修改公共接口时，优先补充必要的类型注解
- 保持函数职责单一，避免在同一个提交中混入无关重构
- 尽量复用现有工具函数、日志能力和目录结构，保持风格一致

## Shell 脚本规范

- 遵循仓库现有脚本风格，保持结构简洁、可重复执行
- 对路径、变量和命令参数进行必要引用，降低环境差异带来的风险
- 新增脚本时应明确输入、输出和失败行为，避免隐式副作用

# 代码测试

## 运行测试

在提交代码前，请确保本地测试通过。若您需要在源码仓中启动 AgentSDK 或运行测试，建议先完成以下准备：

- 按照 [docs/zh/installation_guide.md](docs/zh/installation_guide.md) 完成 Python 依赖、CANN 依赖和第三方仓库依赖安装，并设置对应环境变量
- 在仓库根目录执行 `pip3 install -e .` 安装本地开发版本。安装完成后会注册命令行入口 `agentic_rl`
- 额外安装测试工具：`pytest`、`pytest-html`、`pytest-cov`
- `script/test.sh` 依赖 `bash` 和 `python3`，建议在 Linux 开发环境中执行

源码启动命令示例：

```bash
agentic_rl --config-path /absolute/path/to/config.yaml
```

统一测试脚本如下：

```bash
bash script/test.sh
```

说明：

- 测试脚本会先安装运行测试所需的 Python 依赖，再执行 `tests/` 目录下的 Python 单元测试。具体依赖请以仓库当前的依赖声明文件和测试脚本中的安装命令为准，例如 `setup.py` 与 `script/test.sh`
- 测试脚本会补充第三方仓库到 `PYTHONPATH`。如果您的第三方仓库安装路径与脚本默认值不一致，请先调整 `script/test.sh` 中的路径配置或手动设置 `PYTHONPATH`
- 测试过程中会生成覆盖率及测试报告，输出目录为 `script/coverage/`。其中 HTML 覆盖率报告默认位于 `script/coverage/html/index.html`，JUnit 报告位于 `script/coverage/final.xml`，HTML 测试报告位于 `script/coverage/final.html`
- 如仅需验证局部改动，也可以使用 `python3 -m pytest -vs tests/<path>` 对指定测试进行快速验证

## 添加测试

- 新功能、缺陷修复和行为变更应尽量补充对应测试
- 测试代码统一放在 `tests/` 目录下，并尽量与源码目录结构保持对应关系
- 测试文件命名建议使用 `test_*.py`，测试函数命名建议使用 `test_*`
- 测试应覆盖主要逻辑分支、边界条件和异常路径
- 对外部服务、模型、分布式环境等依赖，优先通过 mock、桩对象或最小替身降低测试成本
- 提交 PR 时，请确保满足项目既有覆盖率要求：分支覆盖率不低于 60%，行覆盖率不低于 80%

# 文档开发

## 文档路径

如果您的变更影响用户使用方式、接口行为或部署配置，请同步更新相关文档：

- 项目总览与快速入口：[README.md](README.md)
- 中文文档：[docs/zh/](docs\zh)
- Python API 文档：[docs/zh/api_python.md](docs\zh\api_python.md)
- 命令行/API 说明：[docs/zh/command_api.md](docs\zh\command_api.md)
## 文档规范

- 使用准确、简洁、可执行的中文描述
- 命令示例应尽量完整，避免缺失前置条件
- 涉及接口、参数或配置变更时，建议补充输入输出说明
- 涉及用户流程变化时，可补充截图、流程图或示例结果
- 提交前请检查链接、文件路径和示例命令是否有效

# 代码提交规范

## Commit 消息格式

所有提交必须遵循以下格式：

```
<type>: <subject>

<body>
```

## Type（类型）

- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整，不影响功能逻辑
- `refactor`: 重构，既不是 Bug 修复也不是新功能
- `perf`: 性能优化
- `test`: 测试相关
- `chore`: 构建、工具或依赖变更
- `ci`: CI/CD 相关变更

## Subject（主题）

- 使用祈使句，首字母小写
- 不超过 50 个字符
- 不以句号结尾
- 描述“做了什么”，而不是“做了什么改动”

## Body（正文，可选）

- 详细说明变更原因、方案和影响范围
- 说明与之前行为的差异
- 可以多行编写，建议每行不超过 72 个字符

# PullRequest

## 提交前检查清单

在提交 Pull Request 之前，请确认：

- [ ] 代码符合项目编码规范
- [ ] 已添加或更新必要测试
- [ ] 本地相关测试已通过
- [ ] 已更新相关文档、示例或说明
- [ ] 已完成自我审查，删除无关改动
- [ ] Commit 消息清晰且符合规范

## PR 创建流程

1. **创建特性分支**

   ```bash
   git checkout -b feature/your-feature-name
   # 或
   git checkout -b fix/issue-number
   ```

2. **进行开发**

   - 编写代码
   - 添加测试
   - 更新文档
   - 确保代码通过本地测试

3. **提交代码**

   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. **推送到 Fork 仓库**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **创建 Pull Request**

   - 访问 GitCode 仓库页面
   - 点击“Pull Request”或“合并请求”
   - 填写 PR 描述，建议包含问题背景、修改方案、测试方法和影响范围

6. **响应评审意见**

   - 及时回复 Reviewers 和 Committers 的反馈
   - 根据评审意见更新代码并重新提交
   - 保持与主分支同步，及时解决冲突

## PR 最佳实践

1. **保持 PR 小规模**

   - 一次 PR 只解决一个问题或实现一个功能
   - 便于评审和理解
   - 提高合并效率
   - 建议单个 PR 的代码变更量控制在 1000 行以内（含测试）

2. **及时更新**

   - 定期同步上游主分支
   - 及时响应评审意见
   - 保持 PR 活跃

3. **清晰描述**

   - 详细描述变更原因和方案
   - 提供测试方法
   - 如有必要，添加截图、示例或对比结果

## PR 评审与合入规则

### 评审要求

1. **评审人员要求**

   - 评审人员必须熟悉相关代码领域
   - 评审人员不能是 PR 作者本人

2. **评审检查项**

   - ✅ 代码质量和风格
   - ✅ 功能正确性
   - ✅ 测试覆盖率（分支 60%，行 80%）
   - ✅ 文档完整性
   - ✅ 性能影响
   - ✅ 安全性
   - ✅ 向后兼容性

3. **CI 检查要求**

   - ✅ 所有 CI 检查必须通过

4. **无 Block 评论**

   - PR 不能存在未解决的阻塞性问题

### 合入规则

1. **Squash and Merge**

   - 将 PR 的所有提交合并为一个提交
   - 保持主分支历史清晰
   - 提交消息使用 PR 标题

2. **必须满足的条件**

   - ✅ 至少获得 2 位 Maintainer 或 Committer 的 `/lgtm`，以及 1 个 `/approve`

3. **禁止的操作**

   - ❌ 禁止 Force Push 到主分支
   - ❌ 禁止合并自己的 PR，必须经过他人评审

### 合并权限

- **Maintainer**：可以合并任何 PR
- **Committer**：可以合并任何 PR
- **Contributor**：无合并权限，需要等待 Maintainer 或 Committer 合并

# CI 说明

CI 检查项目包括：

- `执行Shell`：CI 内部调用
- `Build_arm`：构建集群管理组件二进制包
- `Build_x86`：构建集群管理组件二进制包
- `build_mindio_arm`：构建 mindio 软件包
- `build_mindio_x86`：构建 mindio 软件包
- `code_check`：编码风格、规范与安全检查
- `anti_poison`：病毒扫描
- `sca`：开源合规检查
- `UT_go`：Go 单元测试
- `UT_cpp`：C++ 单元测试
- `UT_python`：Python 单元测试

任意一项失败，都可以通过详情链接查看具体问题。如果属于 CI 自身故障，请[联系 committer](https://gitcode.com/Ascend/community/blob/master/MindSeriesSDK/sigs/AgentSDK/sig-info.yaml)，或通过评论 `rebuild` 尝试重新构建。

# 社区准则

## 行为准则

我们致力于为所有参与者提供友好、安全和包容的协作环境。参与本项目即表示您同意：

- 尊重不同的观点和经验
- 接受建设性的批评和建议
- 聚焦对项目和社区真正有价值的改进
- 以开放、专业和合作的方式进行沟通

## 沟通渠道

- **Issues**：用于反馈缺陷、提出功能建议和讨论技术问题
- **Pull Requests**：用于代码审查和实现讨论
- **SIG 例会与社区页面**：用于了解项目治理、路线和协作活动

# 许可证

通过向本项目贡献代码，您同意您的贡献将按照项目当前许可证进行授权。更多信息请参见 [LICENSE.md](LICENSE.md)。

# 致谢

感谢您为 AgentSDK 做出的贡献。您的参与有助于持续提升项目的可用性、稳定性和生态价值。

# Special Interest Group

## 工作目标和范围

1. 技术聚焦

   围绕基于 Agentic 的多训练后端支持、强化学习环境兼容接口等功能进行深入研究，推动技术发展并解决实际问题。

2. 促进协作

   通过组织会议、技术分享等方式，促进成员之间的协作和知识共享，提升整体技术水平。

3. 最佳实践

   在技术实现、接口设计和开发流程等方面推动最佳实践，降低协作成本，提升系统兼容性和可维护性。

4. 社区建设

   通过代码贡献、技术分享等方式，培养技术人才，推动社区生态建设。

## 成员列表

[SIG 成员列表](https://gitcode.com/Ascend/community/blob/master/MindSeriesSDK/sigs/AgentSDK/sig-info.yaml)。
