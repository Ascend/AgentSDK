# Version Mapping

## Product Version Information

<a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108__Ref249955742"></a>
<table><tbody><tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row244mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.1.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p246mcpsimp"></a>Product</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.1.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p1684675795511"></a><span id="ph925512229126"><a name="ph925512229126"></a><a name="ph925512229126"></a>Agent SDK</span></p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row255mcpsimp"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.2.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p257mcpsimp"></a>Version</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.2.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p233mcpsimp"></a>26.0.0</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_row7259721105019"><th class="firstcol" valign="top" width="25%" id="mcps1.1.3.3.1"><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p7260182135013"></a>Version Type</p>
</th>
<td class="cellrowborder" valign="top" width="75%" headers="mcps1.1.3.3.1 "><p id="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a><a name="zh-cn_topic_0000001938532254_zh-cn_topic_0000001935094108_p72606219501"></a>Beta version</p>
</td>
</tr>
</tbody>
</table>

## Related Product Version Mapping

| Product      | Version     |
|------------|----------|
| Ascend HDK | 26.0.RC1 |
| CANN       | 9.0.0    |

## Virus Scan Results

Virus scan passed.

# Version Compatibility

- Agent SDK: This version has no compatibility issues.

**Table 1** Software version compatibility

| MindSDK Version     | MindSDK Version to Upgrade | CANN Version Compatibility               | Ascend HDK Version Compatibility                  |
|------------------|--------------|--------------------------|-----------------------------------|
| Agent SDK 26.0.0 | N/A         | <li>CANN 9.0.0 and its patch versions</li>| <li>Ascend HDK 26.0.RC1 and its patch versions</li>|

> [!NOTE]NOTE
> Software version compatibility means that when you upgrade the product software version, related software does not need to be upgraded or patched, and existing features remain supported.

# Usage Precautions

None

# Change Description

## New Features

| Feature     | Description                                  | Compatible Product Models             |
|-----------|----------------------------------------|---------------------|
| Agent SDK | Supports fine-tuning the Qwen2.5 7B model for the WebSearcher Agent scenario.| Atlas 800T A2 training server|
| Agent SDK | Supports GRPO training on the GAIA2 dataset.                     | Atlas 800T A2 training server|
| Agent SDK | Integrates with the LangGraph agent development frontend framework.               | Atlas 800T A2 training server|
| Agent SDK | Supports the `verl` training backend engine.                          | Atlas 800T A2 training server|
| Agent SDK | Supports context trajectory compression management and step-level RL training algorithms.         | Atlas 800T A2 training server|

## Interface Changes

**Agent SDK<a name="section047671014474"></a>**

- The new component does not involve interface changes.

## Key Feature Changes

**Agent SDK<a name="zh-cn_topic_0000001935999544_section1641531115220"></a>**

- The new component does not involve key feature changes.

## Resolved Issues

None

## Known Issues<a name="ZH-CN_TOPIC_0000002513525040"></a>

None

# Upgrade Impact

## Impact on the System During the Upgrade

None

## Impact on the System After the Upgrade

None

# 26.0.0 Documentation

| Document                                         | Description                                        | Update Notes                                              |
|-----------------------------------------------|----------------------------------------------|----------------------------------------------------|
| [Agent SDK 26.0.0 User Guide](../../../aura/README.md)| Describes the introduction, installation and deployment, quick start, API reference, and other common operations of Agent SDK.| For details about the changes, see [Agent SDK 26.0.0 User Guide](../../../aura/README.md).|

# Vulnerability Patch List

None
