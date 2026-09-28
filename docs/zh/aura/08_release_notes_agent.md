# 版本说明

## 关键特性

- Aura新增全异步分离、混合批次调度模式训练，新增黑盒Agent特性。
- Aura新增支持qwen3-14b，qwen3.6-27b，qwen3.6-35b-a3b模型。

## 版本配套说明<a name="ZH-CN_TOPIC_0000002545204925"></a>

### 产品版本信息<a name="ZH-CN_TOPIC_0000002513525038"></a>

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>产品名称</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Agent SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>产品版本</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.2.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>版本类型</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release版本</p>
</td>
</tr>
</tbody>
</table>

### 硬件版本配套说明书

**表 1**  Agent SDK硬件版本配套表

| 芯片系列         | 产品示例                     | 架构             |
|--------------|--------------------------|----------------|
| Atlas A2系列产品 | Atlas 800I A2推理服务器       | ARM64 / x86_64 |
| Atlas A3系列产品 | Atlas 900 A3 SuperPoD超节点 | ARM64 / x86_64 |

### 相关产品版本配套说明<a name="ZH-CN_TOPIC_0000002513684954"></a>

**表 2**  Agent SDK软件版本配套表

| Agent SDK | CANN版本 | Ascend HDK版本 |
|-----------|--------|--------------|
| 26.2.0    | 9.0.0  | 26.0.0       |

### 操作系统配套说明

本版本支持 Ubuntu 和 openEuler 操作系统。

## 版本兼容性说明<a name="ZH-CN_TOPIC_0000002545284919"></a>

- Agent SDK：本版本无兼容性问题。

本节表格中"/"表示不可配套，"Y"表示可配套。

**表 3**  Agent SDK与CANN版本兼容

<table style="table-layout: fixed; width: 531px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Agent SDK</th>
    <th colspan="3">CANN版本</th>
  </tr>
  <tr>
    <th>9.0.0</th>
    <th>9.1.0</th>
    <th>9.2.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.2.0</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">/</td>
  </tr>
</tbody>
</table>

**表 4**  Agent SDK与Ascend HDK版本兼容

<table style="table-layout: fixed; width: 531px"><colgroup>
<col style="width: 156px">
<col style="width: 88px">
<col style="width: 91px">
<col style="width: 98px">
<col style="width: 98px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Agent SDK</th>
    <th colspan="3">Ascend HDK版本</th>
  </tr>
  <tr>
    <th>26.0.RC1</th>
    <th>26.1.0</th>
    <th>26.2.0</th>
  </tr></thead>
<tbody>
  <tr>
    <td>26.2.0</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">/</td>
  </tr>
  <tr>
    <td>26.1.0</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">Y</td>
    <td style="text-align: center;">/</td>
  </tr>
</tbody>
</table>

## 版本使用注意事项<a name="ZH-CN_TOPIC_0000002513684952"></a>

无

## 更新说明<a name="ZH-CN_TOPIC_0000002545284923"></a>

### 新增特性<a name="ZH-CN_TOPIC_0000002545284925"></a>

**表 5**  Agent SDK新增特性

| 组件名称 | 特性描述                                              | 配套产品型号                      |
|------|---------------------------------------------------|-----------------------------|
| Aura | 支持[全异步分离模式](./04_user_guide/04_fully_async.md)训练  | Atlas 900 A3 SuperPoD超节点 |
| Aura | 支持[混合批次调度](./04_user_guide/05_mixed_batch.md)训练   | Atlas 900 A3 SuperPoD超节点 |
| Aura | 支持[黑盒Agent](./04_user_guide/07_blackbox_agent.md) | Atlas 900 A3 SuperPoD超节点 |
| Aura | 支持qwen3-14b，qwen3.6-27b，qwen3.6-35b-a3b模型         | Atlas 900 A3 SuperPoD超节点 |

### 业务接口变更<a name="ZH-CN_TOPIC_0000002545204929"></a>

共卡模式新增以下业务接口：

- 配置参数：verl_conf.actor_rollout_ref.rollout 新增 n_gpus_per_node（每节点的NPU卡数）、prompt_length（提示词长度）、response_length（响应长度）。

黑盒 Agent 模式新增以下业务接口：

- 配置参数：新增 agent_engine 配置值 vaee（启用黑盒虚拟引擎模式，Agent 运行于外部独立服务）、agent_proxy_args 配置组（agent_addr、traj_addr、run_id）及 traj_proxy_args.infer_url（轨迹数据拉取地址），用于对接外部 Agent Service 与 TrajProxy。
- 对外接口：新增 AgentProxyClient、TrajProxyClient 类，分别提供 get_agent_response()、get_records_by_session() 方法，封装与外部 Agent Service、TrajProxy 的 HTTP 通信。
- 外部服务接口约定：外部 Agent Service 须暴露 POST /v1/chat/completions 接口，接收 prompt 并返回 session_id 后异步执行 agent loop；TrajProxy 须暴露 GET /trajectory?session_id={session_id} 接口，供训练框架拉取指定 session 的所有 LLM 调用记录。

### 关键特性变更<a name="ZH-CN_TOPIC_0000002513684958"></a>

本版本继承本产品26.1.0版本的所有特性。

### 已解决的问题<a name="ZH-CN_TOPIC_0000002513525034"></a>

无

### 遗留问题<a name="ZH-CN_TOPIC_0000002513525040"></a>

无

## 升级影响<a name="ZH-CN_TOPIC_0000002545284921"></a>

### 升级过程对现行系统的影响<a name="ZH-CN_TOPIC_0000002545204927"></a>

无

### 升级后对现行系统的影响<a name="ZH-CN_TOPIC_0000002513525036"></a>

无

## 26.2.0版本配套文档<a name="ZH-CN_TOPIC_0000002545204931"></a>

**表 6**  Agent SDK 26.2.0版本配套文档

| 文档名称                                          | 内容简介                                         | 更新说明                                               |
|-----------------------------------------------|----------------------------------------------|----------------------------------------------------|
| 《[Agent SDK 26.2.0 用户指南](../../../aura/README.md)》 | 主要包括Agent SDK的简介、安装部署、快速入门、API接口说明以及其他常用的操作。 | 变更详见《[Agent SDK 26.2.0 用户指南](../../../aura/README.md)》。 |

## 病毒扫描结果

病毒扫描通过。

## 漏洞修补列表<a name="ZH-CN_TOPIC_0000002545204923"></a>

无

## 修订记录

**表 7** Agent SDK 版本修订记录

| 文档版本 | 发布日期       | 修改说明    |
|------|------------|---------|
| 01   | 2026-09-23 | 第一次正式发布 |
