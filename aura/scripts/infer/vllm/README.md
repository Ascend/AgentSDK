# 发送推理对话

```shell
# 1. 在 GDE 网页创建任务，通过命令传入 Prefill & Decode 节点及卡数等配置，且根据 vllm_serve.sh 在 GDE 网页设置 并行策略、模型权重 等参数
cd /models/l00619320/code/VLLM_ASCEND_PD_CLUSTER/AgenticRL/scripts/deploy_vllm; sh start_cluster_with_vllm_serve.sh --prefill-instances 1 --decode-instances 2 --prefill-cards-per-instance 8 --decode-cards-per-instance 8 --node-cards 8 --socket-ifname eth0

# 2. 访问 INDEX=0 号容器，推理对话：
curl --location 'http://172.16.8.126:8080/v1/chat/completions' \
--header 'Content-Type: application/json' \
--data '{
  "model": "Qwen3-30B-A3B",
  "messages": [
        {"role": "system", "content": "你是一个能干的助手."},
        {"role": "user", "content": "谁赢得了2020年的世界职业棒球大赛?"}
  ],
  "temperature": 0.5,
  "top_p": 0.9,
  "max_tokens": 100,
  "stream": false,
  "frequency_penalty": 0,
  "presence_penalty": 0
}'
```
