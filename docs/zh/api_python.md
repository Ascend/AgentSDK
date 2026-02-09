# Python接口说明<a name="ZH-CN_TOPIC_0000002459514668"></a>

## 数据类型<a name="ZH-CN_TOPIC_0000002492474277"></a>

### Trajectory<a name="ZH-CN_TOPIC_0000002492474285"></a>

**功能描述<a name="section7608155812579"></a>**

该类用于记录Agent运行的轨迹信息。

**函数原型<a name="section1483104721911"></a>**

```
@dataclass
class Trajectory:
    prompt_tokens: torch.Tensor
    response_tokens: torch.Tensor
    response_masks: torch.Tensor
    idx: int = 0
    trajectory_reward: float | int = 0.0
    chat_completions: list[dict[str, str]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=lambda: {"steps": 0,
                                                             "reward_time": 0.0,
                                                             "env_time": 0.0,
                                                             "llm_time": 0.0,
                                                             "total_time": 0.0})
```

**参数说明<a name="section5277832016"></a>**

|参数名|类型|说明|
|--|--|--|
|idx|int|轨迹的索引或编号，默认为0。|
|prompt_tokens|torch.Tensor|输入提示的token序列，不能包含Nan和Inf。|
|response_tokens|torch.Tensor|模型生成的响应的token序列，不能包含Nan和Inf。|
|response_masks|torch.Tensor|响应token的掩码，用于标识有效的token。不能包含Nan和Inf。|
|trajectory_reward|float \| int|轨迹的奖励值，默认为0.0。|
|chat_completions|list[dict[str,str]]|大模型对话列表，默认为空列表。|
|metrics|dict[str, Any]|轨迹的各项性能指标。其中，Any的取值为：<li>steps：表示轨迹的总执行步数。整数类型，默认为0。<li>reward_time：表示计算奖励值所花费的时间。数值类型，默认为0.0。<li>env_time：表示与环境交互所花费的时间。数值类型，默认为0.0。<li>llm_time：表示LLM推理所花费的时间。数值类型，默认为0.0。<li>total_time：表示整个轨迹执行的总时间。数值类型，默认为0.0。|




## 功能函数参考<a name="ZH-CN_TOPIC_0000002492554185"></a>

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

|属性名|类型|是否必选| 描述| 取值范围|
|--|--|--|--|--|
| server_addresses| List[str] |是| 每个元素为vLLM推理服务的IP地址。即IP:PORT字符串| IP:127.0.0.1; PORT:0-65535 |


#### completions<a name="ZH-CN_TOPIC_0000002468653042"></a>

**功能描述<a name="section76042045113017"></a>**

在BaseEngineWrapper类中为用户提供了推理能力。该属性为一个列表，每个元素均为一个vLLM推理函数。

**函数原型<a name="section171073435330"></a>**

```
result = self.completions[0](completions_request)
```

**参数说明<a name="section946451519388"></a>**

|参数名|类型|是否必选|描述|取值范围|
|--|--|--|--|--|
|prompt|str|是|大语言模型的输入提示。|非空字符串。|
|n|int|否|rollout的次数。|取值必须大于等于1，默认为1。|
|top_k|int|否|文本生成过程中的采样策略参数。即表示在每一步预测时，模型只考虑概率最高的前k个候选词。|取值必须大于等于0，默认为50。|
|logprobs|int|否|输出概率最高的n个候选词的对数概率。|取值大于等于0，默认为1。其中：<li>0：表示禁止输出对数概率。<li>其他值：表示输出对数概率。|
|min_p|float|否|最小概率阈值，用于控制文本生成过程中候选词的概率下限。通过该阈值，可以实现以下操作：<li>过滤掉概率低于min_p的候选词。即确保生成的文本中只保留概率在阈值以上的候选词。<li>通过调整min_p的值，可以控制生成结果的确定性或多样性。|默认为0.0，取值范围：[0.0, 1.0]。|
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

```
initialize()
```


#### generate\_agent\_trajectories\_async<a name="ZH-CN_TOPIC_0000002501573019"></a>

**功能描述<a name="section142771553125211"></a>**

基于Agent执行引擎异步生成Agent轨迹。

**函数原型<a name="section179371655641"></a>**

```
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

```
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




