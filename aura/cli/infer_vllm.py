#!/usr/bin/env python3
# coding=utf-8
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
import argparse
import os
from typing import Any, Dict, List, Optional

import torch
import vllm.distributed.parallel_state as ps

from third_party.rl.mindspeed_rl.models.rollout.vllm_engine import VLLMInferEngine
from third_party.rl.mindspeed_rl.utils.loggers import Loggers

logger = Loggers(
    name="vllm_engine_inference",
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    # Inference arguments group
    group = parser.add_argument_group(title="inference args")
    group.add_argument("--tokenizer-name-or-path", type=str,
                       help="Hugging config path.")
    group.add_argument(
        "--load-format", type=str,
        choices=["auto", "megatron"], default="auto",
        help="Vllm weight load format, support auto from huggingface and from megatron format.")
    group.add_argument("--load", type=str,
        default=None,
        help="Vllm weight path for megatron load format.")
    group.add_argument("--tensor-parallel-size", type=int,
        default=1,
        help="infer tensor parallel size")
    group.add_argument("--query", type=str,
        default="Write an essay about the importance of higher education.",
        help="Input query")
    group.add_argument("--task", type=str,
        choices=["generation", "chat"], default="chat",
        help="Inference task, generation or chat.")
    group.add_argument("--gpu-memory-utilization", type=float, default=0.9,
        help="Device memory utilization for vllm.")

    group = parser.add_argument_group(title="distributed")
    group.add_argument("--distributed-backend", default="nccl",
        choices=['nccl', 'gloo'],
        help="Distributed training/inference backend.")
    group.add_argument("--local-rank", type=int, default=int(os.getenv('LOCAL_RANK', '0')),
        help="Local rank of the process in distributed training.")
    group.add_argument("--prompt-type", type=str, default=None,
        choices=['default', 'empty', 'trl', 'qwen', 'qwen_rl', 'qwen_math_r1', 'llama3', 'mistral', 'mixtral', 'gemma', 'llama2',
         'alpaca', 'deepseek2', 'deepseek2-lite', 'minicpm3', 'cpm', 'baichuan2', 'deepseek3'],
        help='Which template to use for constructing prompt in training/inference. e.g., "qwen"')
    group.add_argument("--prompt-type-path", type=str, default=None,
        help="Path to the json file of templates.")

    group = parser.add_argument_group(title="sampling params")
    group.add_argument("--num-completions", type=int, default=1,
        help="Number of output sequences to return for each input prompt.")
    group.add_argument("--logprobs", type=int, default=1,
        help="Number of log probabilities returned per output token")
    group.add_argument("--max-tokens", type=int, default=128,
        help="Maximum number of tokens generated per output sequence")
    group.add_argument("--top-p", type=float, default=1.0,
        help="Cumulative probability threshold for nucleus sampling")
    group.add_argument("--top-k", type=int, default=-1,
        help="Number of highest probability tokens to consider during sampling, -1 means all tokens")
    group.add_argument("--temperature", type=float, default=1.0,
        help="Temperature parameter controlling sampling randomness")

    return parser.parse_args()


def process_outputs(outputs):
    res = ""
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        res = res + f"Prompt: {prompt!r}\nGenerated Text: {generated_text!r}\n"
    res = res + "-" * 80
    return res


def main():
    logger.info("Starting VLLM Inference Engine")
    args = get_args()

    sampling_config = {
        "num_completions": args.num_completions,
        "logprobs": args.logprobs,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "detokenize": True,
    }

    inference_engine = VLLMInferEngine(
        megatron_config=None,
        sampling_config=sampling_config,
        train_expert_parallel_size=1,
        infer_expert_parallel_size=1,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        prompt_type=args.prompt_type,
        prompt_type_path=args.prompt_type_path,
        train_tensor_parallel_size=args.tensor_parallel_size,
        train_pipeline_parallel_size=1,
        infer_tensor_parallel_size=args.tensor_parallel_size,
        infer_pipeline_parallel_size=1,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        load_format=args.load_format
    )
    
    if args.load_format == "megatron":
        tp_rank = ps._TP.rank_in_group
        weights_path = os.path.join(args.load, f"iter_0000001/mp_rank_{tp_rank:02}/model_optim_rng.pt")

        actor_weights = torch.load(weights_path)['model']
        actor_weights = replace_state_dict_name(
            actor_weights,
            vllm_dict=inference_engine.model.state_dict(),
            arch=inference_engine.model.__class__.__name__)
        logger.info("sync_model_weights")
        inference_engine.sync_model_weights(actor_weights)

        logger.info("init_cache_engine")
        inference_engine.init_cache_engine()

    if args.task == "chat":
        chat_task(inference_engine, args.query)
    elif args.task == "generation":
        generate_task(inference_engine, args.query)


def chat_task(inference_engine: Any, query: str) -> None:
    conversation = [
        {"role": "user", "content": query}
    ]
    outputs = inference_engine.chat(conversation)
    res = process_outputs(outputs)
    logger.info(f"Query: {query}")
    logger.info(f"Responses: \n{res}")


def generate_task(inference_engine: Any, query: str) -> None:
    outputs = inference_engine.llm.generate(
        prompts=[query],
        sampling_params=inference_engine.sampling_params,
    )
    res = process_outputs(outputs)
    logger.info(f"Query: {query}")
    logger.info(f"Responses: \n{res}")


def replace_state_dict_name(state_dict: Dict[str, torch.Tensor], 
                           vllm_dict: Dict[str, torch.Tensor], 
                           arch: Optional[str] = None) -> Dict[str, torch.Tensor]:
    params_mapping = [
        ("embedding.word_embeddings", "model.embed_tokens"),
        ("self_attention.linear_qkv", "self_attn.qkv_proj"),
        ("self_attention.linear_proj", "self_attn.o_proj"),
        ("input_layernorm", "input_layernorm"),
        ("pre_mlp_layernorm", "post_attention_layernorm"),
        ("mlp.linear_fc1", "mlp.gate_up_proj"),
        ("mlp.linear_fc2", "mlp.down_proj"),
        ("decoder.final_layernorm", "model.norm"),
        ("output_layer", "lm_head"),
        # Deepseek add
        ("self_attention.linear_qb", "self_attn.q_b_proj"),
        ("self_attention.linear_kvb", "self_attn.kv_b_proj"),
        ("mlp.router.weight", "mlp.gate.weight"),
        ("mlp.router.expert_bias", "mlp.gate.e_score_correction_bias"),
        ("mlp.shared_experts.linear_fc1", "mlp.shared_experts.gate_up_proj"),
        ("mlp.shared_experts.linear_fc2", "mlp.shared_experts.down_proj"),
        ("mlp.experts.weight1", "mlp.experts.w13_weight"),
        ("mlp.experts.weight2", "mlp.experts.w2_weight"),
        ("self_attention.q_layernorm", "self_attn.q_a_layernorm"),
        ("self_attention.k_layernorm", "self_attn.kv_a_layernorm"),
    ]
    
    new_state_dict = {}
    for name, loaded_weight in state_dict.items():
        if "_extra_state" in name:
            continue
        if "Deepseek" in arch:
            name = _replace_name_m2v_deepseek(name, params_mapping)
        else:
            name = _replace_name_m2v(name, params_mapping)
        
        if "e_score_correction_bias" in name:
            loaded_weight = loaded_weight.to(vllm_dict[name].dtype)
        if "mlp.experts" in name:
            loaded_weight = loaded_weight.view(vllm_dict[name].shape)
        
        new_state_dict[name] = loaded_weight
    return new_state_dict


def _replace_name_m2v(name: str, name_mapping: List[tuple]) -> str:
    for m_name, v_name in name_mapping:
        if m_name not in name:
            continue
            
        if "layers" in name:
            name = name.replace("decoder", "model")
            name_list = name.split(".")

            if "layer_norm_weight" in name_list or "layer_norm_bias" in name_list:
                param_name_list = name_list[:3]
                param_name_list.append(v_name)
                param_name = ".".join(param_name_list)
            else:
                param_name_list = name_list[:3]
                weight_or_bias = name_list[-1]
                param_name_list.append(v_name)
                param_name_list.append(weight_or_bias)
                param_name = ".".join(param_name_list)
            return param_name
        else:
            param_name = name.replace(m_name, v_name)
            return param_name
    return name


def _replace_name_m2v_deepseek(name: str, name_mapping: List[tuple]) -> str:
    for m_name, v_name in name_mapping:
        if m_name not in name:
            continue
            
        if "layers" in name:
            # Replace decoder with model
            name = name.replace("decoder", "model")
        
        param_name = name.replace(m_name, v_name)
        return param_name
    
    return name


if __name__ == "__main__":
    main()
