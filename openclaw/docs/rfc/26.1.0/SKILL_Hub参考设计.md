# 1. 概述

## 1.1 简介

SKILL作为OpenClaw能力扩展的核心机制，当前面临供应链投毒风险与分发管理困难。本提案提供一套完整的SKILL Hub参考设计，支持企业内网自建私有SKILL库，实现安全的技能发现、安装、更新与完整性验证。通过抽象接口隔离各平台（GitHub、Gitee、GitCode）细节，采用适配器模式支持平台扩展，构建可信、可控的SKILL分发体系。

## 1.2 动机

当前SKILL依赖外部社区共享，存在以下核心痛点：

- **供应链安全风险**：企业用户无法审计SKILL来源与安全性，存在隐蔽后门、数据窃取等安全隐患
- **质量参差不齐**：社区SKILL缺乏安全审计机制，质量无法保证
- **私有化部署缺失**：无企业内部SKILL库私有化部署能力，无法满足企业合规要求
- **更新策略缺失**：用户无法自定义SKILL更新策略与可信度验证机制

**用户案例**：

- 某金融企业需要在内网环境使用OpenClaw，但无法连接外部GitHub仓库，亟需内网SKILL库方案
- 某科技公司要求所有使用的SKILL必须经过安全审计与签名验证，当前机制无法满足

**不做此提案的影响**：

- 企业级用户因安全与合规问题无法采用OpenClaw，限制商业化推广
- SKILL生态质量无法保障，劣质或恶意SKILL损害用户体验与安全
- 企业重复造轮子，各自维护私有SKILL分发方案，造成生态碎片化

## 1.3 目标

**目标：**

- 提供私有SKILL库（SKILL Hub）部署方案，支持企业内网自建
- 实现技能的标准化安装/卸载/更新，支持远程仓库与本地路径双模式
- 提供完整性验证机制（SHA256/SHA512/MD5校验、GPG签名验证），保障供应链安全
- 定义统一接口规范，实现多平台（GitHub、Gitee、GitCode）支持

**非目标：**

- 不提供中心化SKILL市场/商店（如VSCode Marketplace），定位为私有化部署参考方案
- 不涉及SKILL开发阶段的代码审查与质量评估
- 不提供SKILL运行时的沙箱隔离与权限控制（由安全沙箱特性覆盖）

---

# 2. 用例分析

## 2.1 用例1：企业内网SKILL Hub部署

**场景描述：** 企业IT管理员在内网服务器部署SKILL Hub，为内部用户提供可信的SKILL源。

**功能点：**

- 配置同步上游仓库（GitHub/Gitee/GitCode）的策略
- 设置访问控制与权限管理
- SKILL元数据管理与分类浏览
- 本地SKILL包上传与内部发布
- 多源优先级管理，支持内外网SKILL源灵活切换

**安全隐私要求：**

- 访问认证（Token/API Key）
- SKILL签名验证强制启用
- 操作审计日志

**DFX要求：**

- **可维护性**：支持配置文件热加载，无需重启服务
- **可扩展性**：SKILL存储支持本地文件系统/对象存储
- **可靠性**：服务异常时不影响已安装SKILL的正常使用

## 2.2 用例2：技能发现与安装

**场景描述：** 开发者或Agent在OpenClaw环境中搜索并安装所需SKILL。

**功能点：**

- 多维度搜索：名称、标签、分类、作者
- SKILL详情查看：版本历史、依赖关系、使用说明
- 远程仓库安装：指定版本或最新版
- 本地路径安装：开发调试场景
- 依赖自动解析与级联安装
- Monorepo模式支持：单仓库管理多个SKILL

**DFX要求：**

- **兼容性**：支持OpenClaw
- **可测试性**：提供安装失败的详细日志与诊断信息

## 2.3 用例3：版本管理与更新

**场景描述：** 系统管理员管理已安装SKILL的版本，执行升级或回退操作。

**功能点：**

- 查看已安装SKILL列表及版本信息
- 升级至最新版本或指定版本
- 卸载SKILL

**安全隐私要求：**

- 版本升级时强制验证签名
- 更新操作需用户确认（可配置）

**DFX要求：**

- **可维护性**：清理旧版本保留策略可配置

## 2.4 用例4：完整性验证

**场景描述：** 在SKILL安装和更新过程中，验证SKILL包的完整性，防止供应链攻击。

**功能点：**

- 下载后自动计算并比对SHA256/SHA512/MD5 checksum
- 支持GPG签名验证
- 可信来源/作者白名单校验

**安全隐私要求：**

- 弱算法（MD5）可配置禁用
- 验证过程日志完整记录

---

# 3. 方案设计

## 3.1 总体方案

SKILL Hub参考设计采用分层架构，核心组件包括：

```text
┌────────────────────────────────────────────────────────────┐
│                    SKILL Hub 架构                            │
├───────────────┬───────────────┬───────────────────────────┤
│   发现层       │    管理层      │        验证层              │
│  Discovery    │   Management  │    Verification           │
│  - 搜索       │   - 安装      │    - Checksum             │
│  - 浏览       │   - 卸载      │    - GPG签名               │
│  - 元数据     │   - 升级      │    - 证书链                │
├───────────────┴───────────────┴───────────────────────────┤
│              统一接口层 (Unified Interface)                  │
│          ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│          │  GitHub │  │  Gitee  │  │ GitCode │            │
│          │ Adapter │  │ Adapter │  │ Adapter │            │
│          └─────────┘  └─────────┘  └─────────┘            │
├────────────────────────────────────────────────────────────┤
│              存储层 (Storage Layer)                         │
│       本地FS / 对象存储  +  元数据数据库                     │
└────────────────────────────────────────────────────────────┘
```

**核心设计思路：**

1. **适配器模式**：抽象统一接口隔离平台差异，新平台只需实现适配器即可接入
2. **工厂模式**：根据仓库URL自动创建对应平台的适配器实例
3. **插件化验证**：验证模块可扩展，支持自定义验证策略
4. **本地优先**：企业内网场景下优先使用本地缓存，减少外网依赖

**技术平台选择：**

- **开发语言**：Python 3.10+
- **存储**：本地文件系统（默认）/ 对象存储
- **API协议**：RESTful API + JSON

## 3.2 技术选型

**方案对比：**

| 方案 | 优势 | 劣势 | 结论 |
|------|------|------|------|
| 集中式Hub服务 | 统一管理，便于审计 | 单点故障，部署复杂 | 企业级推荐 |
| 分布式直连仓库 | 简单轻量，无中间件 | 无法内网隔离，管理困难 | 放弃 |
| 混合模式 | 兼顾灵活与管理 | 实现复杂度高 | **社区版推荐** |
| 集成NPM/PyPI生态 | 复用成熟生态 | 与OpenClaw SKILL格式不兼容 | 放弃 |

**混合模式（社区版推荐）说明：**

- Agent可直接连接外部仓库（GitHub/Gitee/GitCode）发现SKILL
- 同时支持连接企业内部部署的SKILL Hub获取审核后的SKILL
- 通过配置优先级实现内外网SKILL源的灵活切换

## 3.3 功能与性能设计

### 3.3.1 统一接口设计

**核心接口定义（适配器模式）：**

```python
class SkillRepositoryAdapter(ABC):
    """SKILL仓库适配器抽象基类"""
    
    @abstractmethod
    def search(self, query: str, filters: dict) -> List[SkillMetadata]:
        """搜索SKILL"""
        pass
    
    @abstractmethod
    def fetch(self, skill_name: str, version: str) -> SkillPackage:
        """获取SKILL包"""
        pass
    
    @abstractmethod
    def get_versions(self, skill_name: str) -> List[str]:
        """获取版本列表"""
        pass
    
    @abstractmethod
    def verify(self, skill_name: str, version: str, 
               checksum: str = None, signature: str = None) -> bool:
        """验证SKILL包完整性"""
        pass
```

**工厂类实现：**

```python
class RepositoryFactory:
    """根据URL自动创建对应平台的适配器"""
    
    _adapters = {
        "github.com": GitHubAdapter,
        "gitee.com": GiteeAdapter,
        "gitcode.com": GitCodeAdapter,
    }
    
    @classmethod
    def create(cls, repo_url: str) -> SkillRepositoryAdapter:
        parsed_url = urlparse(repo_url)
        hostname = parsed_url.hostname
        if not hostname:
            raise UnsupportedRepositoryError(f"无效的仓库URL: {repo_url}")
        for domain, adapter_class in cls._adapters.items():
            if hostname == domain or hostname.endswith('.' + domain):
                return adapter_class(repo_url)
        raise UnsupportedRepositoryError(f"不支持的仓库类型: {repo_url}")
```

### 3.3.2 SKILL安装流程设计

```text
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 解析SKILL  │ -> │ 解析依赖   │ -> │ 下载包    │ -> │ 完整性   │ -> │ 安装部署  │
│ 名称/版本  │    │ 关系图    │    │ 文件     │    │ 验证     │    │ 到目录    │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
     │                │                │                │                │
     ▼                ▼                ▼                ▼                ▼
   错误处理          循环依赖检测       并发下载          失败则删除        注册元数据
   模糊匹配          版本冲突检测       断点续传          告警记录          清理旧版本
```

**依赖解析规则：**

- 支持语义化版本约束：`^1.0.0`、`>=1.2.0 <2.0.0`、`~1.2.3`
- 循环依赖检测与报错
- 版本冲突时优先使用最高兼容版本，或提示用户选择

**安装模式支持：**

- **Archive模式**：下载zip/tar归档包，解压后安装
- **Contents API模式**：通过平台API递归下载目录内容（适用于Monorepo场景）
- **本地路径模式**：从本地文件系统直接安装，解析SKILL.md获取元数据

### 3.3.3 完整性验证流程设计

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ 下载SKILL包   │ -> │ 计算本地Hash  │ -> │ 比对远程Hash  │ -> │ GPG签名验证  │
│             │    │ (SHA-256)     │    │             │    │ (如启用)      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                   │                   │
                          ▼                   ▼                   ▼
                    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                    │ 记录计算结果  │    │ 不匹配则拒绝  │    │ 验证失败拒绝  │
                    │             │    │ 安装并告警    │    │ 安装并告警    │
                    └─────────────┘    └─────────────┘    └─────────────┘
```

### 3.3.4 性能设计

安全SKILL参考设计需满足以下性能要求：

- **搜索响应**：SKILL搜索应在可接受时间内返回结果
- **安装性能**：单SKILL安装流程应高效完成，含依赖解析
- **并发处理**：支持多Agent同时安装，避免资源竞争
- **缓存加速**：本地缓存机制减少重复网络请求
- **依赖解析**：复杂依赖树解析应在合理时间内完成

## 3.4 安全隐私与DFX设计

### 3.4.1 安全设计

- **传输安全**：所有通信使用HTTPS/TLS
- **访问控制**：API Key认证 + IP白名单（可选）
- **完整性保护**：SHA-256校验 + GPG签名双重验证
- **审计追踪**：所有安装/卸载/更新操作记录审计日志
- **最小权限**：Hub服务以非root用户运行
- **安全权限**：安装后自动设置目录755/文件644权限

### 3.4.2 隐私设计

- 不收集用户Agent的运行数据或业务数据
- 仅记录必要的操作日志（SKILL名称、版本、操作类型、时间戳）

### 3.4.3 DFX设计

**兼容性：**

- 适配器层屏蔽平台差异，Agent侧无感知
- 支持OpenClaw
- 存储层抽象，支持无缝切换存储后端

**可维护性：**

- 适配器独立实现，新增平台不影响已有适配器
- 配置文件YAML格式，易于理解与修改
- 提供管理CLI工具，简化运维操作

**可测试性：**

- 适配器层提供Mock实现，便于单元测试
- 提供测试SKILL仓库，覆盖正常/异常场景
- 集成测试覆盖完整安装/卸载/升级流程

**可靠性：**

- 安装过程原子性：失败自动回滚，不残留中间文件
- 网络异常支持自动重试
- Hub服务故障时不影响已安装SKILL的正常使用

## 3.5 编程与调用设计

### 3.5.1 编程模型基本设计

**开发环境设计：**

- **硬件环境**：x86_64 / aarch64 服务器或边缘设备
- **软件环境**：Python 3.10+
- **依赖管理**：requirements.txt 限定版本

**开发约束：**

- 适配器实现需遵循统一接口规范
- 第三方依赖需经过安全评估

**可验收设计：**

- 提供端到端测试脚本（安装→验证→卸载完整流程）

### 3.5.2 接口定义与设计

#### 3.5.2.1 skill.hub.search（SKILL搜索接口）

**接口描述：** 在SKILL Hub或远程仓库中搜索SKILL

**接口原型：**

```python
def search(query: str = None,
           category: str = None,
           tags: List[str] = None,
           source: str = "all",
           page: int = 1,
           page_size: int = 20) -> SearchResult:
```

**输入参数：**

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
|----------|-----------|------|------|----------|
| query | 输入 | str | 搜索关键词（名称/描述模糊匹配） | 任意字符串 |
| category | 输入 | str | SKILL分类过滤 | 预定义分类 |
| tags | 输入 | List[str] | 标签过滤 | 预定义标签列表 |
| source | 输入 | str | 搜索范围 | "all" / "local" / "remote" / 仓库URL |
| page | 输入 | int | 分页页码 | ≥ 1，默认1 |
| page_size | 输入 | int | 每页数量 | 1-100，默认20 |

**返回参数：**

| 参数名称 | 类型 | 描述 | 取值范围 |
|----------|------|------|----------|
| total | int | 总结果数 | ≥ 0 |
| skills | list | SKILL列表 | 每项包含name/version/description/author/tags |
| page | int | 当前页码 | ≥ 1 |
| page_size | int | 每页数量 | 1-100 |

**调用参考代码：**

```python
from openclaw.skill_hub import search

# 搜索安全类SKILL
results = search(query="security", category="security")
for skill in results.skills:
    print(f"{skill.name} v{skill.version} by {skill.author}")

# 从企业内部Hub搜索
results = search(query="data", source="https://hub.mycompany.com")
```

#### 3.5.2.2 skill.hub.install（SKILL安装接口）

**接口描述：** 安装指定SKILL，支持远程仓库和本地路径

**接口原型：**

```python
def install(skill_ref: str,
            version: str = "latest",
            source: str = None,
            verify: bool = True,
            force: bool = False) -> InstallResult:
```

**输入参数：**

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
|----------|-----------|------|------|----------|
| skill_ref | 输入 | str | SKILL引用（名称或路径） | 必填，如 "security/baseline" |
| version | 输入 | str | 指定版本 | 语义化版本号或 "latest"，默认"latest" |
| source | 输入 | str | 安装源 | 仓库URL或本地路径，默认读取配置 |
| verify | 输入 | bool | 是否验证完整性 | true / false，默认true |
| force | 输入 | bool | 强制重新安装 | true / false，默认false |

**返回参数：**

| 参数名称 | 类型 | 描述 | 取值范围 |
|----------|------|------|----------|
| skill_name | str | 安装的SKILL名称 | - |
| version | str | 安装版本 | 语义化版本号 |
| install_path | str | 安装路径 | 本地绝对路径 |
| dependencies | list | 安装的依赖列表 | [{name, version}, ...] |
| verify_result | dict | 验证结果 | {checksum_verified, signature_verified} |

**异常处理：**

- `SkillNotFoundError`：SKILL不存在，错误码3001
- `VersionNotFoundError`：指定版本不存在，错误码3002
- `VerifyError`：完整性验证失败，错误码3003
- `DependencyError`：依赖解析/安装失败，错误码3004

**调用参考代码：**

```python
from openclaw.skill_hub import install

# 安装最新版
result = install("security/baseline")
print(f"安装至: {result.install_path}")

# 安装指定版本（从企业Hub）
result = install("security/baseline", 
                 version="1.2.0",
                 source="https://hub.mycompany.com")

# 本地路径安装（开发调试）
result = install("./my-skill", source="/home/dev/projects/my-skill")
```

#### 3.5.2.3 skill.hub.uninstall（SKILL卸载接口）

**接口描述：** 卸载已安装的SKILL，可选清理未使用的依赖

**接口原型：**

```python
def uninstall(skill_name: str,
              remove_unused_deps: bool = False,
              backup: bool = True) -> UninstallResult:
```

**输入参数：**

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
|----------|-----------|------|------|----------|
| skill_name | 输入 | str | 要卸载的SKILL名称 | 必填 |
| remove_unused_deps | 输入 | bool | 是否移除未使用的依赖 | true / false，默认false |
| backup | 输入 | bool | 卸载前备份 | true / false，默认true |

**返回参数：**

| 参数名称 | 类型 | 描述 | 取值范围 |
|----------|------|------|----------|
| skill_name | str | 卸载的SKILL名称 | - |
| removed_deps | list | 移除的依赖列表 | [{name, version}] |
| backup_path | str | 备份路径（如启用备份） | 本地绝对路径或null |

#### 3.5.2.4 skill.hub.update（SKILL更新接口）

**接口描述：** 更新已安装的SKILL至新版本

**接口原型：**

```python
def update(skill_name: str = None,
           version: str = "latest",
           all_skills: bool = False) -> List[UpdateResult]:
```

**输入参数：**

| 参数名称 | 输入/输出 | 类型 | 描述 | 取值范围 |
|----------|-----------|------|------|----------|
| skill_name | 输入 | str | 要更新的SKILL名称 | 默认null |
| version | 输入 | str | 目标版本 | "latest" 或指定版本 |
| all_skills | 输入 | bool | 更新所有SKILL | true / false，默认false |

**返回参数：**

| 参数名称 | 类型 | 描述 | 取值范围 |
|----------|------|------|----------|
| updates | list | 更新结果列表 | [{skill_name, old_version, new_version, status}] |
| failed | list | 失败的更新 | [{skill_name, error}] |

---

# 4. 缺点和风险

## 4.1 潜在风险

| 风险项 | 风险描述 | 影响等级 | 应对措施 |
|--------|----------|----------|----------|
| 供应链攻击 | 恶意SKILL通过验证机制绕过 | 高 | 多层验证（checksum+签名+人工审核） |
| 仓库不可用 | 外部仓库（GitHub等）访问受限 | 中 | 本地缓存机制，企业Hub镜像同步 |
| 版本冲突 | 依赖版本冲突导致安装失败 | 中 | 清晰的错误提示，依赖冲突解决方案指南 |
| 存储膨胀 | 多版本缓存导致磁盘空间不足 | 低 | 自动清理策略，保留版本数可配置 |
| 兼容性断裂 | SKILL新版本不兼容旧配置 | 中 | 强制语义化版本，升级前兼容性检查 |

## 4.2 负面影响

- **首次部署复杂度**：企业部署SKILL Hub需要额外的运维投入
- **网络延迟**：完整性验证增加了安装耗时
- **存储开销**：本地缓存与多版本保留增加了磁盘占用

## 4.3 实现成本

- **开发工作量**：需要投入开发资源完成核心功能实现
- **测试验证**：需进行充分的测试验证
- **维护成本**：需持续投入资源进行适配器维护与问题修复

## 4.4 兼容性考虑

- **API版本**：SKILL Hub API遵循语义化版本，保持向后兼容
- **Agent兼容性**：需OpenClaw Agent v1.0+ 支持
- **仓库格式**：适配器屏蔽平台差异，Agent侧无需感知
- **升级路径**：旧版本Agent逐步迁移至新Hub，提供兼容性说明

---

# 5. 现有技术

## 5.1 参考项目

### 5.1.1 VSCode Extension Marketplace

- **借鉴点**：SKILL发现、搜索、安装的交互流程设计
- **差异点**：Marketplace为中心化服务，本提案聚焦私有化部署

### 5.1.2 Helm Chart Repository

- **借鉴点**：基于HTTP服务器的包仓库设计、索引文件格式
- **差异点**：Helm专注Kubernetes生态，本提案面向通用SKILL管理

### 5.1.3 Python PyPI / npm Registry

- **借鉴点**：包版本管理、依赖解析、完整性验证机制
- **差异点**：PyPI/npm为特定语言生态，本提案跨语言、跨平台

### 5.1.4 OCI Registry (Docker Registry V2)

- **借鉴点**：基于内容寻址的存储、分层设计
- **差异点**：OCI专注容器镜像，本提案面向SKILL包管理

## 5.2 技术差异优势

| 维度 | 传统包管理器 | OpenClaw SKILL Hub |
|------|-------------|-------------------|
| 部署模式 | 中心化公共仓库 | 支持私有化部署 |
| 平台支持 | 单一平台 | 多平台适配器（GitHub/Gitee/GitCode） |
| 验证机制 | 单一checksum | 多层验证（checksum + GPG签名） |
| 生态集成 | 特定语言/工具 | 与OpenClaw Agent深度集成 |
| 管理粒度 | 包级别 | SKILL + 依赖 + 版本全生命周期 |

---

# 6. 未解决问题

1. **企业审核流程**：SKILL上架企业Hub的审核流程与质量标准需社区讨论
2. **GPG密钥管理**：签名验证的公钥分发与更新机制
3. **SKILL评分体系**：是否引入社区评分/下载量等质量指标

---
