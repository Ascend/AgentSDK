# 样例和指导<a name="ZH-CN_TOPIC_0000002492554189"></a>

## AgentEngine对接示例<a name="ZH-CN_TOPIC_0000002459514660"></a>

使用AgenticRL框架时，开发者需自行实现一个继承自BaseEngineWrapper的子类，以完成与自定义代理引擎的对接。本文档提供了一个模拟示例，帮助开发者快速进行调试和验证。

**实现自定义MockEngineWrapper<a name="section142771553125211"></a>**

1.  创建mock\_engine\_wrapper.py文件

    ```
    vi /your_forder/mock_engine_wrapper.py
    ```

2.  自定义MockEngineWrapper类

    ```
    import random
    import torch
    from typing import List, Any, Dict
    from agentic_rl import BaseEngineWrapper, Trajectory
    
    class MockEngineWrapper(BaseEngineWrapper):
        def __init__(
                self,
                agent_name: str,
                tokenizer: Any,
                sampling_params: Dict[str, Any],
                max_prompt_length=128*1024,
                max_response_length=8*1024,
                n_parallel_agents=8,
                max_steps=5
        ):
            super().__init__(agent_name, tokenizer, sampling_params, max_prompt_length, max_response_length,
                             n_parallel_agents, max_steps)
    
        def initialize(self):
            pass
    
        def generate_agent_trajectories_async(
                self, tasks
        ) -> List[Trajectory]:
    
            if not isinstance(tasks, list):
                raise TypeError("tasks must be a list")
    
            if not tasks:
                raise ValueError("tasks list cannot be empty")
    
            for task in tasks:
                if not isinstance(task, dict):
                    raise TypeError("tasks must be a list of dictionary")
    
            mock_results = {
                "prompt_tokens": torch.tensor([101, 200, 300, 400], dtype=torch.long),
                "response_tokens": torch.tensor([500, 600, 700], dtype=torch.long),
                "response_masks": torch.tensor([1, 1, 1], dtype=torch.long),
                "trajectory_reward": random.uniform(-1, 1),
                "idx": random.randint(0, 9999),
                "chat_completions": [
                    {"role": "assistant", "content": "This is a mock response."}
                ],
                "metrics": {
                    "steps": 3,
                    "reward_time": 0.01,
                    "env_time": 0.05,
                    "llm_time": 0.02,
                    "total_time": 0.08,
                },
            }
            trajectories = [Trajectory(**mock_results) for _ in tasks]
            return trajectories
    ```

**修改配置文件<a name="section17641024165316"></a>**

```
vi /your_config_directory_path/your_config_file_name.yaml
```

修改Agent引擎路径字段

```
# ...
agent_engine_wrapper_path: "/your_forder/mock_engine_wrapper.py"
# ...
```

**启动训练任务<a name="section11265124225512"></a>**

```
agentic_rl --config-path /your_config_directory_path/your_config_file_name.yaml
```


