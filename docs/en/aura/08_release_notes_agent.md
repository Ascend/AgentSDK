# Release Notes

## Key Features

- Aura adds training in the fully asynchronous disaggregation mode and the mixed batch scheduling mode, and adds the black-box agent feature.
- Aura adds support for the Qwen3-14B, Qwen3.6-27B, and Qwen3.6-35B-A3B models.

## Version Mapping<a name="ZH-CN_TOPIC_0000002545204925"></a>

### Product Version Information<a name="ZH-CN_TOPIC_0000002513525038"></a>

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>Product Name</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Agent SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>Product Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.2.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Release version</p>
</td>
</tr>
</tbody>
</table>

### Hardware Version Mapping

**Table 1**  Agent SDK hardware version mapping

| Chip Series | Product Example | Architecture |
|--------------|--------------------------|----------------|
| Atlas A2 product | Atlas 800I A2 | ARM64 / x86_64 |
| Atlas A3 product | Atlas 900 A3 SuperPoD supernode | ARM64 / x86_64 |

### Related Product Version Mapping<a name="ZH-CN_TOPIC_0000002513684954"></a>

**Table 2**  Agent SDK software version mapping

| Agent SDK | CANN Version | Ascend HDK Version |
|-----------|--------|--------------|
| 26.2.0    | 9.0.0  | 26.0.0       |

### Operating System Mapping

This version supports the Ubuntu and openEuler operating systems.

## Version Compatibility<a name="ZH-CN_TOPIC_0000002545284919"></a>

- Agent SDK: This version has no compatibility issues.

In the tables in this section, "/" indicates that the versions are not compatible, and "Y" indicates that they are compatible.

**Table 3**  Agent SDK and CANN version compatibility

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
    <th colspan="3">CANN Version</th>
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

**Table 4**  Agent SDK and Ascend HDK version compatibility

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
    <th colspan="3">Ascend HDK Version</th>
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

## Usage Precautions<a name="ZH-CN_TOPIC_0000002513684952"></a>

None

## Update Notes<a name="ZH-CN_TOPIC_0000002545284923"></a>

### New Features<a name="ZH-CN_TOPIC_0000002545284925"></a>

**Table 5**  Agent SDK new features

| Component | Feature Description | Compatible Product Model |
|------|---------------------------------------------------|-----------------------------|
| Aura | Supports training in the [fully asynchronous disaggregation mode](./04_user_guide/04_fully_async.md) | Atlas 900 A3 SuperPoD supernode |
| Aura | Supports training with [mixed batch scheduling](./04_user_guide/05_mixed_batch.md) | Atlas 900 A3 SuperPoD supernode |
| Aura | Supports the [black-box agent](./04_user_guide/07_blackbox_agent.md) | Atlas 900 A3 SuperPoD supernode |
| Aura | Supports the Qwen3-14B, Qwen3.6-27B, and Qwen3.6-35B-A3B models | Atlas 900 A3 SuperPoD supernode |

### Service Interface Changes<a name="ZH-CN_TOPIC_0000002545204929"></a>

Co-located mode adds the following service interfaces:

- Configuration parameters: `verl_conf.actor_rollout_ref.rollout` adds `n_gpus_per_node` (number of NPUs per node), `prompt_length` (prompt length), and `response_length` (response length).

Black-box agent mode adds the following service interfaces:

- Configuration parameters: A new `agent_engine` configuration value `vaee` (enables the black-box virtual engine mode, in which the agent runs in an external standalone service), the `agent_proxy_args` configuration group (`agent_addr`, `traj_addr`, and `run_id`), and `traj_proxy_args.infer_url` (the address for pulling trajectory data), which are used to connect to the external agent service and TrajProxy.
- External interfaces: New `AgentProxyClient` and `TrajProxyClient` classes, which provide the `get_agent_response()` and `get_records_by_session()` methods, respectively, and encapsulate the HTTP communication with the external agent service and TrajProxy.
- External service interface conventions: The external agent service must expose the `POST /v1/chat/completions` endpoint, receive a prompt, and return a `session_id`, after which it runs the agent loop asynchronously. TrajProxy must expose the `GET /trajectory?session_id={session_id}` endpoint so that the training framework can pull all LLM call records of a specified session.

### Key Feature Changes<a name="ZH-CN_TOPIC_0000002513684958"></a>

This version inherits all features of version 26.1.0 of this product.

### Resolved Issues<a name="ZH-CN_TOPIC_0000002513525034"></a>

None

### Known Issues<a name="ZH-CN_TOPIC_0000002513525040"></a>

None

## Upgrade Impact<a name="ZH-CN_TOPIC_0000002545284921"></a>

### Impact on the System During the Upgrade<a name="ZH-CN_TOPIC_0000002545204927"></a>

None

### Impact on the System After the Upgrade<a name="ZH-CN_TOPIC_0000002513525036"></a>

None

## 26.2.0 Documentation<a name="ZH-CN_TOPIC_0000002545204931"></a>

**Table 6**  Agent SDK 26.2.0 documentation

| Document | Description | Update Notes |
|-----------------------------------------------|----------------------------------------------|----------------------------------------------------|
| [Agent SDK 26.2.0 User Guide](../../../aura/README.md) | Mainly describes the introduction to Agent SDK, installation and deployment, quick start, API reference, and other common operations. | For details about the changes, see the [Agent SDK 26.2.0 User Guide](../../../aura/README.md). |

## Virus Scan Results

Virus scan passed.

## Vulnerability Patch List<a name="ZH-CN_TOPIC_0000002545204923"></a>

None

## Revision History

**Table 7** Agent SDK revision history

| Document Version | Release Date | Change Description |
|------|------------|---------|
| 01   | 2026-09-23 | First official release |
