# Python接口说明<a name="ZH-CN_TOPIC_0000002459514668"></a>

> 注意：AgentSDK可通过Python接口进行应用开发，从代码调用角度上来说所有Python侧接口都可以被调用。本章节仅列出业务提供的对外接口，其余未进行说明的接口用户请勿直接调用。

## 数据类型<a name="ZH-CN_TOPIC_0000002492474277"></a>

### Trajectory<a name="ZH-CN_TOPIC_0000002492474285"></a>

**功能描述<a name="section7608155812579"></a>**

该类用于记录Agent运行的轨迹信息。

**函数原型<a name="section1483104721911"></a>**

```python
@dataclass
class Trajectory:
    prompt_tokens: torch.Tensor
    response_tokens: torch.Tensor
    response_masks: torch.Tensor
    idx: int = 0
    trajectory_reward: float | int = 0.0
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=lambda: {"steps": 0,
                                                             "toolcall_reward": 0.0,
                                                             "res_reward": 0.0,
                                                             "reward_time": 0.0,
                                                             "env_time": 0.0,
                                                             "llm_time": 0.0,
                                                             "total_time": 0.0})
```

**参数说明<a name="section5277832016"></a>**

|参数名|类型|说明|
|--|--|--|
|prompt_tokens|torch.Tensor|输入提示的token序列，不能包含Nan和Inf。|
|response_tokens|torch.Tensor|模型生成的响应的token序列，不能包含Nan和Inf。|
|response_masks|torch.Tensor|响应token的掩码，用于标识有效的token。不能包含Nan和Inf。|
|idx|int|轨迹的索引或编号，默认为0。|
|trajectory_reward|float \| int|轨迹的奖励值，默认为0.0。|
|chat_completions|list[dict[str,str]]|大模型对话列表，默认为空列表。|
|metrics|dict[str, Any]|轨迹的各项性能指标。其中，Any的取值为：<li>steps：表示轨迹的总执行步数。整数类型，默认为0。</li><li>reward_time：表示计算奖励值所花费的时间。数值类型，默认为0.0。</li><li>toolcall_reward：表示轨迹生成过程中工具调用奖励值。数值类型，默认为0.0。</li><li>res_reward：表示最终答案的奖励值。数值类型，默认为0.0。</li><li>env_time：表示与环境交互所花费的时间。数值类型，默认为0.0。</li><li>llm_time：表示LLM推理所花费的时间。数值类型，默认为0.0。</li><li>total_time：表示整个轨迹执行的总时间。数值类型，默认为0.0。</li>|

### Step<a name="ZH-CN_TOPIC_0000002492474285"></a>

**功能描述<a name="section7608155812579"></a>**

该类用于记录step-level模式下Agent运行的单轨迹信息。

**函数原型<a name="section1483104721911"></a>**

```python
@dataclass
class Step:
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    thought: str = ""
    action: Any = None
    observation: Any = None
    model_response: str = ""
    info: dict = field(default_factory=dict)
    reward: float = 0.0
    done: bool = False
    mc_return: float = 0.0

```

**参数说明<a name="section5277832016"></a>**

**Step类参数说明**

|参数名|类型| 说明                                             |
|--|--|------------------------------------------------|
|chat_completions|list[dict[str, str]]| 推理所有的完整对话上下文（含历史轮次），用于构造模型输入。                  |
|thought|str| 模型回复中 `<think>` 标签内的内容，表示模型在本步骤的内部推理。          |
|action|Any| 模型回复中 `<tool call>` 标签内的内容，表示模型决定执行的动作（如工具调用）。 |
|observation|Any| 本步骤接收到的外部观测：第 0 轮为用户原始提问，后续轮次为上一轮动作的执行结果（如工具返回）。 |
|model_response|str| 大模型生成的完整回复内容（即 `'role': 'assistant'` 的 `content`）。 |
|info|dict| 附加信息字典，默认为空，可用于记录工具 ID、耗时等元数据。                 |
|reward|float| 本步骤获得的即时奖励，默认为 `0.0`，反映当前动作的质量。                |
|done|bool| 是否在本步骤终止轨迹，默认为 `False`，标识任务是否完成。               |
|mc_return|float| 从本步骤开始的 Monte Carlo 回报，默认为 `0.0`，用于策略梯度训练。     |


### StepTrajectory<a name="ZH-CN_TOPIC_0000002492474285"></a>

**功能描述<a name="section7608155812579"></a>**

该类用于记录step-level模式下Agent运行的完整轨迹信息。

**函数原型<a name="section1483104721911"></a>**

```python
@dataclass
class StepTrajectory(Trajectory):
    task: Any = None
    steps: list[Step] = field(default_factory=list)
```

**参数说明<a name="section5277832016"></a>**

**StepTrajectory类参数说明**

|参数名|类型| 说明                                    |
|---|--|---------------------------------------|
|task|Any| 原始任务输入（如用户问题），默认为 `None`，作为整个轨迹的初始目标。 |
|steps|list[Step]| 第 i 个Step包含从第 1 轮到第 i+1 轮的完整对话上下文。    |

## 功能函数参考<a name="ZH-CN_TOPIC_0000002492554185"></a>

### MemoryConfig<a name="ZH-CN_TOPIC_0000002468333046"></a>

#### 类描述<a name="ZH-CN_TOPIC_0000002516415135"></a>

**功能描述<a name="section8910541196"></a>**

MemoryConfig类管理内存配置。包括： 去think压缩（思维简化），摘要生成和上下文窗口管理以及聊天和嵌入的模型端点。

**参数说明<a name="section2028115207207"></a>**

| 参数名                        | 类型     | 说明                                                                                                                                 | 取值                                                                 |
|----------------------------|--------|------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|
| simplify_thinking          | bool   | 是否从消息中简化思考内容。                                                                                                                      | 默认为False。                                                          |
| use_summary                | bool   | 启用对话历史的自动摘要功能。                                                                                                                     | 默认为False。                                                          |
| max_summary_length         | int    | 生成摘要的最大长度（以token为单位）。                                                                                                              | 默认为1024，取值范围：[1, max_prompt_length]。                               |
| max_prompt_length          | int    | 包括上下文在内的提示词的最大总长度（以token为单位）。                                                                                                      | 默认为8192，取值范围：[1, 128K]。                                            |
| before_raw_message         | int    | 保留头部不受影响的初始消息数量，范围内的消息不受摘要/思考内容简化的影响。                                                                                              | 取值必须大于等于0，默认为0。                                                    |
| end_raw_message            | int    | 保留尾部不受影响的初始消息数量，范围内的消息不受摘要/思考内容简化的影响。                                                                                              | 取值必须小于等于0，默认为0。                                                    |
| summary_system_prompt      | str    | 用于生成摘要的系统提示模板。                                                                                                                     | 固定字符串，具体值如下实例所示。                                                   |
| oai_client                 | OpenAI | 用于摘要生成的OpenAI客户端。                                                                                                                  | 默认为空。                                                              |
| oai_model_name             | str    | 用于摘要生成的OpenAI模型名称。                                                                                                                 | 默认为qwen2.5-7b-instruct。                                            |
| train_model_tokenizer_path | str    | 用于计算上下文大小的tokenizer路径。                                                                                                             | 默认为空字符串。                                                           |
| model_config               | dict   | 利用Pydantic的能力在赋值时提供校验。validate_assignment为True确保对象创建后的完整性，在赋值后会检查该值是否符合字段的类型和约束条件；arbitrary_types_allowed为True表示允许任意类型的数据作为模型的字段值。 | 默认为{"validate_assignment": True, "arbitrary_types_allowed": True}。 |

summary_system_prompt具体模板如下：

```markdown
Please act as a summarization assistant to provide a precise and comprehensive summary of the specified content. The summary should meet the following requirements:
1. **Core Objective**: Extract the core information of the content, retain all key details (such as important data, viewpoints, time, characters, conclusions, etc.), and remove redundant information to ensure the summary is both concise and complete.
2. **Structural Requirements**:
- Begin with 1-2 sentences summarizing the content theme or core conclusion;
- List key information (such as main viewpoints, important events, data indicators, etc.) in bullet points, each containing specific details (avoid general statements);
- Conclude with the impact of the content, follow-up recommendations, or unresolved issues (if applicable).
3. **Detail Specifications**:
- Retain key terms, numerical values, and proper nouns (such as names of people, places, institutions) from the original text without alteration or abbreviation;
- If the content involves a timeline, organize key events in chronological order;
- If the content includes multiple viewpoints, clearly distinguish the positions of different entities;
- Avoid adding personal interpretations to maintain the objectivity of the summary.
4. **Length Recommendation**: The summary should not exceed {max_summary_length} tokens.
Please summarize the following dialogue content based on the above requirements.
```

### MemorySummary<a name="ZH-CN_TOPIC_0000002468333046"></a>

#### 类描述<a name="ZH-CN_TOPIC_0000002516415135"></a>

**功能描述<a name="section8910541196"></a>**

MemorySummary类提供自动对话摘要的内存管理，继承于MemorySimple，当上下文超出配置限制时，提供自动摘要功能。

**参数说明<a name="section2028115207207"></a>**

| 参数名         | 类型            | 说明                 | 取值     |
|-------------|---------------|--------------------|--------|
| chat_client | SummaryClient | 用于摘要生成的OpenAI客户端。  | 默认为空。  |

#### get\_prompt\_messages<a name="ZH-CN_TOPIC_0000002501573019"></a>

**功能描述<a name="section142771553125211"></a>**

获取格式化消息以便生成提示并自动进行摘要。

**函数原型<a name="section179371655641"></a>**

```python
get_prompt_messages(config: dict | None = None) -> List[dict]
```

**参数说明<a name="section868816556614"></a>**

| 参数名    | 数据类型 | 可选/必选 | 描述   |
|--------|------|-------|------|
| config | dict | 可选    | 配置参数 |

类MemorySummary定义的方法get\_prompt\_messages其输入参数config待更新的配置字典或对象属性。

**返回值说明<a name="section246545519364"></a>**

List\[dict\]是一个非空列表，列表中的每个元素均为字典，形如：{"role": "user", "content": "hello"}，其中role的取值范围是["system", "user", "assistant", "tool", "summary"], content为非空字符串。
具体示例如下：

| 键        | 描述       |
|----------|----------|
| role     | 产生上下文的角色 |
| content  | 具体上下文信息  |

**示例<a name="section1556916138163"></a>**

```python
class MemorySummary(MemorySimple):

    @validate_params(
        config=dict(
            validator=lambda x: isinstance(x, dict) or x is None, message="config must be a dictionary or None"
        )
    )
    def get_prompt_messages(self, config: dict | None = None) -> list[dict]:
       
        if config is not None:
            self.update_config(config)

        messages = self._get_effective_messages()

        if self.config.use_summary and self._is_overlength(messages):
            logger.info("Prompt length exceeds max_prompt_length, triggering summarization.")
            messages = self._handle_overlength()

        final_messages = self._format_summary_message(messages)

        final_messages = self._apply_thinking_filter(final_messages)

        if self._is_overlength(final_messages):
            logger.warning(
                f"PROMPT_TRUNCATION: Prompt length ({self._get_total_length(final_messages)}) "
                f"exceeds max_prompt_length ({self.config.max_prompt_length})"
            )

        return MemorySimple._remove_message_other_key(final_messages)
```

### BaseEngineWrapper<a name="ZH-CN_TOPIC_0000002468333046"></a>

#### 类描述<a name="ZH-CN_TOPIC_0000002516415135"></a>

**功能描述<a name="section8910541196"></a>**

BaseEngineWrapper类提供统一的抽象接口，允许不同的AgentEngine自行适配，从而实现AgenticRL框架与多种类型AgentEngine的功能对接。

**参数说明<a name="section2028115207207"></a>**

|参数名|类型|说明|取值|
|--|--|--|--|
|agent_name|str|Agent场景名称。|可输入的字符串中只能包含字母、数字和下划线，且不能以数字开头，不能为空。|
|tokenizer|object|文本分词器对象。|不能为空，且必须为PreTrainedTokenizer或PreTrainedTokenizerFast类型。|
|sampling_params|dict|模型推理时的采样参数。|默认为空。|
|max_prompt_length|int|输入提示的最大长度。|默认为128K，取值范围：[1, 128K]。|
|max_response_length|int|输出响应的最大长度。|默认为8K，取值范围： [1, 8K]。|
|n_parallel_agents|int|并行执行的Agent数量。|默认为8，取值范围：[1, 64]。|
|max_steps|int|Agent执行的最大步骤数。|默认为5，取值范围： [1, 10]。|

#### server_addresses

**功能描述**

在BaseEngineWrapper类中为用户提供了推理服务的本地IP地址。该属性为一个列表，每个元素均为一个vLLM推理服务的IP地址字符串。

**属性说明**

|属性名|类型|是否必选| 描述                                                                                                                                           | 取值范围                           |
|--|--|--|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------|
| server_addresses| List[str] |是| 每个元素为vLLM推理服务的IP地址，即IP:PORT字符串。<br/>该属性内的IP地址可用于访问openai兼容的RESTful接口，包括“v1/completions”和“v1/chat/completions”，接口的请求格式和响应格式均为兼容openai格式的JSON。 | IP:127.0.0.1<br/>PORT:0-65535。 |

##### **v1/completions**

**参数说明**

|参数名|类型|是否必选|描述|取值范围|
|--|--|--|--|--|
|prompt|str|是|大语言模型的输入提示。|非空字符串。|
|n|int|否|rollout的次数。|取值必须大于等于1，默认为1。|
|top_k|int|否|文本生成过程中的采样策略参数。即表示在每一步预测时，模型只考虑概率最高的前k个候选词。|取值必须大于等于0，默认为50。|
|logprobs|int|否|输出概率最高的n个候选词的对数概率。|取值大于等于0，默认为1。其中：<li>0：表示禁止输出对数概率。</li><li>其他值：表示输出对数概率。</li>|
|min_p|float|否|最小概率阈值，用于控制文本生成过程中候选词的概率下限。通过该阈值，可以实现以下操作：<li>过滤掉概率低于min_p的候选词。即确保生成的文本中只保留概率在阈值以上的候选词。</li><li>通过调整min_p的值，可以控制生成结果的确定性或多样性。</li>|默认为0.0，取值范围：[0.0, 1.0]。|
|detokenize|bool|否|是否进行反分词。|默认为False，即不进行反分词。取值为False或True。|
|frequency_penalty|float|否|频率惩罚。对高频候选词施加与出现次数成正比的惩罚，作用于整个生成文本。|默认为0.0，取值范围：[-2.0, 2.0]。|
|max_tokens|int|否|允许生成的最大候选词个数。实际值受配置限制。|默认为128，取值范围：[1, 64000]。|
|min_tokens|int|否|每个输出序列所需的最小候选词个数。实际值受配置限制。|默认为0，取值范围：[0, 64000]。|
|presence_penalty|float|否|存在惩罚。对已出现过的候选词施加固定惩罚，作用于所有已出现的候选词，不区分频率。|默认为0.0，取值范围：[-2.0, 2.0]。|
|seed|int|否|随机种子，用于生成确定性输出。|取值范围：[-65535, 65535]。|
|temperature|float|否|采样温度，取值越大输出越随机。|默认为0.2，取值范围：[0.0, 2.0]。|
|top_p|float|否|核采样，仅考虑概率质量在top_p内的候选词。|[1e-8, 1.0]|

**示例**

```python
import aiohttp

base_url = f"http://{self.server_addresses[0]}/v1/completions"   # 使用 v1/completions 接口
headers = {"Content-Type": "application/json"}

async with aiohttp.ClientSession() as session:
    async with session.post(base_url, headers=headers, json=completions_request) as response:
        if response.status != 200:
            err_msg = await response.text()
            raise Exception(f"Http request failed. status = {response.status}: {err_msg}") 
        result = await response.json()
```

##### **v1/chat/completions**

**参数说明**

| 参数名               | 类型                  |是否必选|描述| 取值范围                                                                                                                          |
|-------------------|---------------------|--|--|-------------------------------------------------------------------------------------------------------------------------------|
| messages          | list[dict[str,str]] |是|大语言模型的输入。| 非空列表，列表中的每个元素均为字典，形如：{"role": "user", "content": "hello"}，其中role的取值范围是["system", "user", "assistant", "tool"], content为非空字符串。 |
| n                 | int                 |否|rollout的次数。| 取值必须大于等于1，默认为1。                                                                                                               |
| top_k             | int                 |否|文本生成过程中的采样策略参数。即表示在每一步预测时，模型只考虑概率最高的前k个候选词。| 取值必须大于等于0，默认为50。                                                                                                              |
| logprobs          | int                 |否|输出概率最高的n个候选词的对数概率。| 取值大于等于0，默认为1。其中：<li>0：表示禁止输出对数概率。</li><li>其他值：表示输出对数概率。</li>                                                                            |
| min_p             | float               |否|最小概率阈值，用于控制文本生成过程中候选词的概率下限。通过该阈值，可以实现以下操作：<li>过滤掉概率低于min_p的候选词。即确保生成的文本中只保留概率在阈值以上的候选词。</li><li>通过调整min_p的值，可以控制生成结果的确定性或多样性。| 默认为0.0，取值范围：[0.0, 1.0]。 </li>                                                                                                      |
| detokenize        | bool                |否|是否进行反分词。| 默认为False，即不进行反分词。取值为False或True。                                                                                               |
| frequency_penalty | float               |否|频率惩罚。对高频候选词施加与出现次数成正比的惩罚，作用于整个生成文本。| 默认为0.0，取值范围：[-2.0, 2.0]。                                                                                                      |
| max_tokens        | int                 |否|允许生成的最大候选词个数。实际值受配置限制。| 默认为128，取值范围：[1, 64000]。                                                                                                       |
| min_tokens        | int                 |否|每个输出序列所需的最小候选词个数。实际值受配置限制。| 默认为0，取值范围：[0, 64000]。                                                                                                         |
| presence_penalty  | float               |否|存在惩罚。对已出现过的候选词施加固定惩罚，作用于所有已出现的候选词，不区分频率。| 默认为0.0，取值范围：[-2.0, 2.0]。                                                                                                      |
| seed              | int                 |否|随机种子，用于生成确定性输出。| 取值范围：[-65535, 65535]。                                                                                                         |
| temperature       | float               |否|采样温度，取值越大输出越随机。| 默认为0.2，取值范围：[0.0, 2.0]。                                                                                                       |
| top_p             | float               |否|核采样，仅考虑概率质量在top_p内的候选词。| [1e-8, 1.0]                                                                                                                   |

**示例**

```python
import aiohttp

base_url = f"http://{self.server_addresses[0]}/v1/chat/completions"   # 使用 v1/chat/completions 接口
headers = {"Content-Type": "application/json"}

async with aiohttp.ClientSession() as session:
    async with session.post(base_url, headers=headers, json=completions_request) as response:
        if response.status != 200:
            err_msg = await response.text()
            raise Exception(f"Http request failed. status = {response.status}: {err_msg}") 
        result = await response.json()
```

#### completions<a name="ZH-CN_TOPIC_0000002468653042"></a>

**功能描述<a name="section76042045113017"></a>**

在BaseEngineWrapper类中为用户提供了推理能力。该属性为一个列表，每个元素均为一个vLLM推理函数。

**函数原型<a name="section171073435330"></a>**

```python
result = self.completions[0](completions_request)
```

**参数说明<a name="section946451519388"></a>**

|参数名|类型|是否必选|描述|取值范围|
|--|--|--|--|--|
|prompt|str|是|大语言模型的输入提示。|非空字符串。|
|n|int|否|rollout的次数。|取值必须大于等于1，默认为1。|
|top_k|int|否|文本生成过程中的采样策略参数。即表示在每一步预测时，模型只考虑概率最高的前k个候选词。|取值必须大于等于0，默认为50。|
|logprobs|int|否|输出概率最高的n个候选词的对数概率。|取值大于等于0，默认为1。其中：<li>0：表示禁止输出对数概率。</li><li>其他值：表示输出对数概率。</li>|
|min_p|float|否|最小概率阈值，用于控制文本生成过程中候选词的概率下限。通过该阈值，可以实现以下操作：<li>过滤掉概率低于min_p的候选词。即确保生成的文本中只保留概率在阈值以上的候选词。</li><li>通过调整min_p的值，可以控制生成结果的确定性或多样性。|默认为0.0，取值范围：[0.0, 1.0]。</li>|
|detokenize|bool|否|是否进行反分词。|默认为False，即不进行反分词。取值为False或True。|
|frequency_penalty|float|否|频率惩罚。对高频候选词施加与出现次数成正比的惩罚，作用于整个生成文本。|默认为0.0，取值范围：[-2.0, 2.0]。|
|max_tokens|int|否|允许生成的最大候选词个数。实际值受配置限制。|默认为128，取值范围：[1, 64000]。|
|min_tokens|int|否|每个输出序列所需的最小候选词个数。实际值受配置限制。|默认为0，取值范围：[0, 64000]。|
|presence_penalty|float|否|存在惩罚。对已出现过的候选词施加固定惩罚，作用于所有已出现的候选词，不区分频率。|默认为0.0，取值范围：[-2.0, 2.0]。|
|seed|int|否|随机种子，用于生成确定性输出。|取值范围：[-65535, 65535]。|
|temperature|float|否|采样温度，取值越大输出越随机。|默认为0.2，取值范围：[0.0, 2.0]。|
|top_p|float|否|核采样，仅考虑概率质量在top_p内的候选词。|[1e-8, 1.0]|

#### initialize<a name="ZH-CN_TOPIC_0000002502198633"></a>

**功能描述<a name="section142771553125211"></a>**

执行AgentEngine必要的初始化流程，具体功能由其派生类自行实现。

**函数原型<a name="section179371655641"></a>**

```python
initialize()
```

#### generate\_agent\_trajectories\_async<a name="ZH-CN_TOPIC_0000002501573019"></a>

**功能描述<a name="section142771553125211"></a>**

基于Agent执行引擎异步生成Agent轨迹。

**函数原型<a name="section179371655641"></a>**

```python
generate_agent_trajectories_async(tasks: List[dict]) -> List[Trajectory]
```

**参数说明<a name="section868816556614"></a>**

|参数名|数据类型|可选/必选|描述|
|--|--|--|--|
|tasks|List[dict]|必选|任务列表|

抽象类BaseEngineWrapper定义的抽象方法generate\_agent\_trajectories\_async提供统一的接口模板，具体实现逻辑由各派生类根据自身特性定义，其输入参数tasks是由Agent SDK构造的任务列表，列表中每个字典元素的各个字段如下：

|键|描述|
|--|--|
|id|任务编号|
|question|当前任务的问题内容|
|ground_truth|该任务的正确答案|

**返回值说明<a name="section246545519364"></a>**

|数据类型|说明|
|--|--|
|List[Trajectory]|生成的Agent轨迹序列，其中每一个元素均为Trajectory类型的对象.|

**示例<a name="section1556916138163"></a>**

```python
from agentic_rl import BaseEngineWrapper, Trajectory

class MockEngineWrapper(BaseEngineWrapper):
    def initialize(self):
        print("Initializing mock engine...")

    def generate_agent_trajectories_async(self, tasks):
        return [Trajectory(idx=0, prompt_tokens=torch.tensor([7, 8, 9]), response_tokens=torch.tensor([10, 11, 12]),
                           response_masks=torch.tensor([1, 1, 1]), trajectory_reward=2.0,
                           chat_completions=[{"role": "assistant", "content": "test"}],
                           metrics={"steps": 1, "reward_time": 2.0, "env_time": 3.0, "llm_time": 4.0, "total_time": 9.0})]

MockEngine = MockEngineWrapper(agent_name="mock_agent_name",
                               tokenizer=...,               # 文本分词器对象
                               sampling_params={"mock":"sampling_params"},
                               max_prompt_length=128*1024,
                               max_response_length=8*1024,
                               n_parallel_agents=16,
                               max_steps=8)
trajectories = MockEngine.generate_agent_trajectories_async()
```
