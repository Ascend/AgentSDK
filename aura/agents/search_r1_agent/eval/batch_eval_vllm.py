#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

import transformers
import random
import requests
import re
import string
import json
import os
import math
import time
from multiprocessing import Process
from collections.abc import Mapping, Sequence
import argparse

# ================= 引入 vLLM =================
from vllm import LLM, SamplingParams


# ================= Reward / Scoring 代码 (保持不变) =================
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) < 1:
        return None
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0.0, score=1.0):
    answer = extract_solution(solution_str=solution_str)
    # 为了避免并发打印混乱，这里建议减少打印或只在特定情况打印
    # print(f"answer is {solution_str} and ground_truth is {ground_truth}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score


# ================= 评分方式选择器 =================
SCORE_METHODS = {
    'em': compute_score_em,
    'subem': compute_score_subem,
}


def get_score_func(method='subem'):
    """获取评分函数

    Args:
        method: 评分方式，'em' 表示精确匹配，'subem' 表示子字符串匹配

    Returns:
        对应的评分函数
    """
    if method not in SCORE_METHODS:
        raise ValueError(f"Unknown score method: {method}. Available: {list(SCORE_METHODS.keys())}")
    return SCORE_METHODS[method]


# ================= 搜索相关 =================
def get_query(text):
    import re

    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def search(query: str):
    try:
        payload = {"queries": [query], "topk": 3, "return_scores": True}
        # 设置超时，防止多进程时某个请求卡死
        response = requests.post("http://10.44.101.107:8000/retrieve", json=payload, timeout=30)
        results = response.json()['result']
    except Exception:
        # 并发环境下偶尔可能会失败，容错处理
        return ""

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    if results:
        return _passages2string(results[0])
    return ""


def read_trivia_qa_validation(data_path):
    from datasets import load_dataset

    dataset = load_dataset("parquet", data_files={"validation": f"{data_path}/validation-*.parquet"})

    val_ds = dataset["validation"]
    print(val_ds[0])
    return val_ds.to_list()


def extract_qa(item):
    # schema 1：prompt / reward_model
    # if (
    #     isinstance(item.get("prompt"), list)
    #     and len(item["prompt"]) > 0
    #     and "content" in item["prompt"][0]
    #     and "reward_model" in item
    #     and "ground_truth" in item["reward_model"]
    # ):
    #     return (
    #         item["prompt"][0]["content"],
    #         item["reward_model"]["ground_truth"]
    #     )
    #
    # # schema 2：parquet trivia_qa
    # # if "question" in item and "answer" in item:
    # #     return (
    # #         item["question"],
    # #         item["answer"].get("normalized_aliases", [])
    # #     )
    # if "question" in item and "answer" in item:
    #     return (
    #         item["question"],
    #         [item["answer"]],
    #     )
    # --------------NQ--------------------------------
    if "question" in item and "reward_model" in item:
        return (item["question"], item["reward_model"]["ground_truth"])

    raise ValueError(f"Unknown data schema: keys={item.keys()}")


# ================= 核心评测逻辑 (Worker内调用) =================
def eval_one_sample(question, ground_truth, tokenizer, llm, score_method='subem'):
    """
    单个样本的评测逻辑，包含 while True 循环

    Args:
        question: 问题
        ground_truth: 标准答案
        tokenizer: tokenizer
        llm: vLLM LLM 实例
        score_method: 评分方式，'em' 或 'subem'
    """
    score_func = get_score_func(score_method)
    question = question.strip()
    if question[-1] != '?':
        question += '?'

    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    prompt_text = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as your want. \
        If you find no further external knowledge worker_main, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    stop_tokens = ["</search>", "</search>\n"]

    if tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False
        )

    # 这里的 prompt 用于累积历史上下文
    prompt = prompt_text

    # 限制最大轮数防止死循环
    max_turns = 100
    current_turn = 0

    while True:
        current_turn += 1
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=4096, stop=stop_tokens, include_stop_str_in_output=True
        )

        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        output_text = outputs[0].outputs[0].text

        is_searching = False
        if "</search>" in output_text:
            is_searching = True

        # 1. 如果没有搜索标签，或者超过最大轮数，强制结束
        if not is_searching or current_turn >= max_turns:
            reward = score_func(output_text, ground_truth)
            return reward, output_text

        # 2. 如果需要搜索，执行 Python 搜索逻辑
        tmp_query = get_query(output_text)

        if tmp_query:
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text


# ================= 批量评测逻辑 (多并发推理) =================
class SampleState:
    """追踪单个样本的评测状态"""

    def __init__(self, idx, item, question, ground_truth, prompt, original_index):
        self.idx = idx  # 在当前批次中的索引
        self.item = item  # 原始数据项
        self.question = question  # 问题
        self.ground_truth = ground_truth  # 标准答案
        self.prompt = prompt  # 当前累积的 prompt
        self.original_index = original_index  # 原始数据索引
        self.current_turn = 0  # 当前轮次
        self.finished = False  # 是否完成
        self.reward = 0  # 得分
        self.output_text = ""  # 最终输出
        self.full_output = ""  # 完整输出历史


def prepare_sample_state(item, idx, tokenizer):
    """准备单个样本的初始状态"""
    question, raw_ground_truth = extract_qa(item)

    if isinstance(raw_ground_truth, Mapping):
        ground_truth_dict = raw_ground_truth
    elif isinstance(raw_ground_truth, str):
        try:
            tmp_data = json.loads(raw_ground_truth)
            ground_truth_dict = {"target": tmp_data}
        except json.decoder.JSONDecodeError:
            ground_truth_dict = {"target": raw_ground_truth}
    elif isinstance(raw_ground_truth, Sequence):
        ground_truth_dict = {"target": list(raw_ground_truth)}
    else:
        raise TypeError(f"Unsupported ground_truth type: {type(raw_ground_truth)}")

    question = question.strip()
    if question[-1] != '?':
        question += '?'

    prompt_text = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as your want. \
        If you find no further external knowledge worker_main, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}], add_generation_prompt=True, tokenize=False
        )

    return SampleState(
        idx=idx,
        item=item,
        question=question,
        ground_truth=ground_truth_dict,
        prompt=prompt_text,
        original_index=item.get('index', idx),
    )


def batch_eval_samples(data_list, tokenizer, llm, batch_size=16, max_turns=100, gpu_id=0, score_method='subem'):
    """
    批量评测多个样本，使用多并发 vLLM 推理加速

    Args:
        data_list: 待评测的数据列表
        tokenizer: tokenizer
        llm: vLLM LLM 实例
        batch_size: 每批并发处理的样本数量
        max_turns: 每个样本的最大搜索轮次
        gpu_id: GPU ID (用于打印日志)
        score_method: 评分方式，'em' 表示精确匹配，'subem' 表示子字符串匹配

    Returns:
        results: 评测结果列表
    """
    score_func = get_score_func(score_method)
    batch_start_time = time.time()
    stop_tokens = ["</search>", "</search>\n"]
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # 耗时统计变量
    total_inference_time = 0.0
    total_search_time = 0.0
    total_search_count = 0
    total_inference_count = 0

    # 初始化所有样本状态
    all_states = []
    for idx, item in enumerate(data_list):
        try:
            state = prepare_sample_state(item, idx, tokenizer)
            all_states.append(state)
        except Exception as e:
            print(f"[Worker {gpu_id}] Error preparing sample {idx}: {e}")
            continue

    # 结果列表
    results = []

    # 待处理的样本队列索引
    pending_queue = list(range(len(all_states)))
    # 当前正在处理的样本状态列表
    active_states = []

    # 填充初始批次
    while len(active_states) < batch_size and pending_queue:
        state_idx = pending_queue.pop(0)
        active_states.append(all_states[state_idx])

    total_processed = 0

    while active_states:
        # 收集所有活跃样本的 prompts
        prompts = [state.prompt for state in active_states]

        # 批量推理
        sampling_params = SamplingParams(
            temperature=0.7, max_tokens=4096, stop=stop_tokens, include_stop_str_in_output=True
        )

        inference_start = time.time()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        inference_time = time.time() - inference_start
        total_inference_time += inference_time
        total_inference_count += 1

        # 处理每个样本的输出
        finished_indices = []
        for i, (state, output) in enumerate(zip(active_states, outputs)):
            state.current_turn += 1
            output_text = output.outputs[0].text
            state.full_output += output_text

            is_searching = "</search>" in output_text

            # 判断是否完成
            if not is_searching or state.current_turn >= max_turns:
                # 样本完成，计算得分
                state.finished = True
                state.output_text = state.full_output
                state.reward = score_func(state.full_output, state.ground_truth)
                finished_indices.append(i)

                # 记录结果
                result_item = {
                    "original_index": state.original_index,
                    "question": state.question,
                    "ground_truth": state.item.get('reward_model', {}).get('ground_truth', ''),
                    "prediction": state.output_text,
                    "score": state.reward,
                }
                results.append(result_item)
                total_processed += 1

                if total_processed % 10 == 0:
                    print(
                        f"[Worker {gpu_id}] Processed {total_processed}/{len(all_states)}, "
                        f"Active: {len(active_states)}, Pending: {len(pending_queue)}"
                    )
            else:
                # 需要继续搜索
                tmp_query = get_query(output_text)
                if tmp_query:
                    search_start = time.time()
                    search_results = search(tmp_query)
                    search_time = time.time() - search_start
                    total_search_time += search_time
                    total_search_count += 1
                else:
                    search_results = ''

                search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
                state.prompt += search_text
                state.full_output += f"<information>{search_results}</information>\n\n"

        # 移除已完成的样本（从后往前移除，避免索引错乱）
        for i in sorted(finished_indices, reverse=True):
            active_states.pop(i)

        # 从待处理队列补充新样本
        while len(active_states) < batch_size and pending_queue:
            state_idx = pending_queue.pop(0)
            active_states.append(all_states[state_idx])

    batch_total_time = time.time() - batch_start_time

    # 打印详细耗时统计
    print(f"\n[Worker {gpu_id}] ------ Batch Eval Timing ------")
    print(f"[Worker {gpu_id}] Total batch time: {batch_total_time:.2f}s")
    print(
        f"[Worker {gpu_id}] Total inference time: {total_inference_time:.2f}s ({total_inference_time / batch_total_time * 100:.1f}%)"
    )
    print(
        f"[Worker {gpu_id}] Total search time: {total_search_time:.2f}s ({total_search_time / batch_total_time * 100:.1f}%)"
    )
    print(f"[Worker {gpu_id}] Inference rounds: {total_inference_count}")
    print(f"[Worker {gpu_id}] Search calls: {total_search_count}")
    if total_inference_count > 0:
        print(f"[Worker {gpu_id}] Avg inference time per round: {total_inference_time / total_inference_count:.3f}s")
    if total_search_count > 0:
        print(f"[Worker {gpu_id}] Avg search time per call: {total_search_time / total_search_count:.3f}s")
    print(f"[Worker {gpu_id}] Samples completed: {len(results)}")
    print(f"[Worker {gpu_id}] ------------------------------\n")

    return results


# ================= Worker 进程入口 =================
def worker_main(gpu_id, subset_data, output_file, model_id, batch_size=16, use_batch_eval=True, score_method='subem'):
    """
    每个进程独立执行的函数

    Args:
        gpu_id: GPU ID
        subset_data: 该进程负责的数据子集
        output_file: 输出文件路径
        model_id: 模型路径
        batch_size: 批量推理的并发数量 (仅 use_batch_eval=True 时生效)
        use_batch_eval: 是否使用批量评测模式 (默认 True)
        score_method: 评分方式，'em' 表示精确匹配，'subem' 表示子字符串匹配
    """
    worker_start_time = time.time()
    print(
        f"[Worker {gpu_id}] Starting... Processing {len(subset_data)} samples. "
        f"Batch mode: {use_batch_eval}, Batch size: {batch_size}, Score method: {score_method}"
    )

    # --- 关键：设置环境变量，让当前进程只看到指定的显卡 ---
    # 如果是 NPU (Ascend)：
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(gpu_id)
    # 如果是 NVIDIA GPU：
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # 每个进程独立加载 Tokenizer 和 模型
        # 注意：trust_remote_code=True
        model_load_start = time.time()
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        # 初始化 vLLM，tensor_parallel_size=1 表示单卡运行
        llm = LLM(
            model=model_id,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=32768,
        )
        model_load_time = time.time() - model_load_start
        print(f"[Worker {gpu_id}] Model loaded in {model_load_time:.2f}s")

        eval_start_time = time.time()
        if use_batch_eval:
            # ========== 批量评测模式 (多并发推理) ==========
            results = batch_eval_samples(
                data_list=subset_data,
                tokenizer=tokenizer,
                llm=llm,
                batch_size=batch_size,
                max_turns=100,
                gpu_id=gpu_id,
                score_method=score_method,
            )

            eval_time = time.time() - eval_start_time

            # 写入结果文件
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for result_item in results:
                    f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")

            worker_total_time = time.time() - worker_start_time
            avg_time_per_sample = eval_time / len(results) if results else 0
            throughput = len(results) / eval_time if eval_time > 0 else 0

            print(f"\n[Worker {gpu_id}] ====== Timing Statistics ======")
            print(f"[Worker {gpu_id}] Model load time: {model_load_time:.2f}s")
            print(f"[Worker {gpu_id}] Evaluation time: {eval_time:.2f}s")
            print(f"[Worker {gpu_id}] Total worker time: {worker_total_time:.2f}s")
            print(f"[Worker {gpu_id}] Samples processed: {len(results)}")
            print(f"[Worker {gpu_id}] Avg time per sample: {avg_time_per_sample:.2f}s")
            print(f"[Worker {gpu_id}] Throughput: {throughput:.2f} samples/s")
            print(f"[Worker {gpu_id}] ================================\n")

        else:
            # ========== 原始逐个评测模式 ==========
            with open(output_file, 'w', encoding='utf-8') as f_out:
                for idx, item in enumerate(subset_data):
                    try:
                        question, raw_ground_truth = extract_qa(item)
                        if isinstance(raw_ground_truth, Mapping):
                            ground_truth_dict = raw_ground_truth

                        elif isinstance(raw_ground_truth, str):
                            try:
                                tmp_data = json.loads(raw_ground_truth)
                                ground_truth_dict = {"target": tmp_data}
                            except json.decoder.JSONDecodeError:
                                ground_truth_dict = {"target": raw_ground_truth}
                        elif isinstance(raw_ground_truth, Sequence):
                            # list / tuple 等，但排除 str（上面已处理）
                            ground_truth_dict = {"target": list(raw_ground_truth)}
                        else:
                            raise TypeError(f"Unsupported ground_truth type: {type(raw_ground_truth)}")

                        reward, output_text = eval_one_sample(
                            question, ground_truth_dict, tokenizer, llm, score_method=score_method
                        )

                        result_item = {
                            "original_index": item.get('index', idx),  # 尽量保留原始数据的索引
                            "question": question,
                            "ground_truth": raw_ground_truth,
                            "prediction": output_text,
                            "score": reward,
                        }
                        f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                        f_out.flush()

                        if idx % 5 == 0:
                            elapsed = time.time() - eval_start_time
                            avg_time = elapsed / (idx + 1) if idx > 0 else elapsed
                            eta = avg_time * (len(subset_data) - idx - 1)
                            print(
                                f"[Worker {gpu_id}] Processed {idx}/{len(subset_data)}, "
                                f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s"
                            )

                    except Exception as e:
                        print(f"[Worker {gpu_id}] Error on item {idx}: {e}")
                        continue

            eval_time = time.time() - eval_start_time
            worker_total_time = time.time() - worker_start_time
            processed_count = len(subset_data)
            avg_time_per_sample = eval_time / processed_count if processed_count > 0 else 0
            throughput = processed_count / eval_time if eval_time > 0 else 0

            print(f"\n[Worker {gpu_id}] ====== Timing Statistics ======")
            print(f"[Worker {gpu_id}] Model load time: {model_load_time:.2f}s")
            print(f"[Worker {gpu_id}] Evaluation time: {eval_time:.2f}s")
            print(f"[Worker {gpu_id}] Total worker time: {worker_total_time:.2f}s")
            print(f"[Worker {gpu_id}] Samples processed: {processed_count}")
            print(f"[Worker {gpu_id}] Avg time per sample: {avg_time_per_sample:.2f}s")
            print(f"[Worker {gpu_id}] Throughput: {throughput:.2f} samples/s")
            print(f"[Worker {gpu_id}] ================================\n")

    except Exception as e:
        print(f"[Worker {gpu_id}] Critical Initialization Error: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM-based Search_R1 Evaluation Script")

    # ===== 数据与输出 =====
    parser.add_argument("--data-file", type=str, default="data/nq_search/test.json", help="Path to evaluation dataset")
    parser.add_argument(
        "--output-file",
        type=str,
        default="eval_results_vllm_nq_subem_OriginalWeight.jsonl",
        help="Path to output jsonl file",
    )

    # ===== 模型配置 =====
    parser.add_argument(
        "--model-id",
        type=str,
        default="/opt/DPC/models/model/Qwen2.5-7B-Instruct",
        help="Model path or HuggingFace model id",
    )
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs used for inference")

    # ===== 批量评测配置 =====
    parser.add_argument("--use-batch-eval", action="store_true", help="Enable batch evaluation mode")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size per inference step")
    parser.add_argument(
        "--score-method", type=str, default="em", choices=["em", "subem"], help="Scoring method: em | subem"
    )

    return parser.parse_args()


def main(args):
    # ================= E2E 计时开始 =================
    e2e_start_time = time.time()

    # ================= 配置参数 =================
    DATA_FILE = args.data_file
    FINAL_OUTPUT_FILE = args.output_file
    model_id = args.model_id

    NUM_GPUS = args.num_gpus
    USE_BATCH_EVAL = args.use_batch_eval
    BATCH_SIZE = args.batch_size
    SCORE_METHOD = args.score_method

    # ================= 参数打印=================
    print("===== Evaluation Configuration =====")
    print(f"Data file       : {DATA_FILE}")
    print(f"Output file     : {FINAL_OUTPUT_FILE}")
    print(f"Model ID        : {model_id}")
    print(f"Num GPUs        : {NUM_GPUS}")
    print(f"Use batch eval  : {USE_BATCH_EVAL}")
    print(f"Batch size      : {BATCH_SIZE}")
    print(f"Score method    : {SCORE_METHOD}")
    print("====================================")
    # ========================

    print("=" * 60)
    print("E2E Evaluation Script Started")
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ===== 数据加载计时 =====
    data_load_start = time.time()
    print(f"Loading data from {DATA_FILE}...")
    print(f"Batch evaluation mode: {USE_BATCH_EVAL}, Batch size: {BATCH_SIZE}, Score method: {SCORE_METHOD}")

    if os.path.isdir(DATA_FILE):
        dataset_raw = read_trivia_qa_validation(DATA_FILE)
    else:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            try:
                dataset_raw = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                dataset_raw = [json.loads(line) for line in f]

    total_samples = len(dataset_raw)
    data_load_time = time.time() - data_load_start
    print(f"Total samples: {total_samples}")
    print(f"Data loading time: {data_load_time:.2f}s")

    # 给数据打上原始索引，方便后续排序
    for i, item in enumerate(dataset_raw):
        item['index'] = i

    # 数据切分
    chunk_size = math.ceil(total_samples / NUM_GPUS)
    chunks = [dataset_raw[i : i + chunk_size] for i in range(0, total_samples, chunk_size)]

    # 确保 chunks 长度等于 GPU 数 (如果数据太少，后面的 chunk 为空)
    while len(chunks) < NUM_GPUS:
        chunks.append([])

    processes = []
    temp_files = []

    # 启动多进程
    for gpu_id in range(NUM_GPUS):
        if not chunks[gpu_id]:
            continue  # 跳过没有数据的进程

        temp_file = f"eval_results_worker_{gpu_id}.jsonl"
        temp_files.append(temp_file)

        # 如果存在旧文件先删除
        if os.path.exists(temp_file):
            os.remove(temp_file)

        p = Process(
            target=worker_main,
            args=(gpu_id, chunks[gpu_id], temp_file, model_id, BATCH_SIZE, USE_BATCH_EVAL, SCORE_METHOD),
        )
        processes.append(p)
        p.start()

    workers_start_time = time.time()
    print(f"Started {len(processes)} worker processes on {NUM_GPUS} GPUs...")

    # 等待所有进程结束
    for p in processes:
        p.join()

    workers_time = time.time() - workers_start_time
    print(f"All workers finished in {workers_time:.2f}s. Merging results...")

    # ===== 结果合并计时 =====
    merge_start_time = time.time()

    # 合并结果
    final_results = []
    total_score = 0
    count = 0

    with open(FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for temp_file in temp_files:
            if not os.path.exists(temp_file):
                continue
            with open(temp_file, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    try:
                        data = json.loads(line)
                        final_results.append(data)
                        total_score += data['score']
                        count += 1
                        f_out.write(line)
                    except:
                        pass
            # 可选：合并后删除临时文件
            # os.remove(temp_file)

    # 按原始索引排序（可选，为了和输入顺序一致）
    final_results.sort(key=lambda x: x.get('original_index', 0))

    merge_time = time.time() - merge_start_time

    # ================= E2E 总耗时统计 =================
    e2e_total_time = time.time() - e2e_start_time

    print("\n" + "=" * 60)
    print("E2E EVALUATION TIMING SUMMARY")
    print("=" * 60)
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print(f"{'Phase':<30} {'Time (s)':<15} {'Percentage':<15}")
    print("-" * 60)
    print(f"{'Data Loading':<30} {data_load_time:<15.2f} {data_load_time / e2e_total_time * 100:<15.1f}%")
    print(f"{'Workers Execution':<30} {workers_time:<15.2f} {workers_time / e2e_total_time * 100:<15.1f}%")
    print(f"{'Results Merging':<30} {merge_time:<15.2f} {merge_time / e2e_total_time * 100:<15.1f}%")
    print("-" * 60)
    print(f"{'E2E Total Time':<30} {e2e_total_time:<15.2f} {'100.0':<15}%")
    print("=" * 60)

    # ===== 性能指标 =====
    if count > 0 and e2e_total_time > 0:
        overall_throughput = count / e2e_total_time
        effective_throughput = count / workers_time if workers_time > 0 else 0
        print("\nPerformance Metrics:")
        print(f"  - Total samples processed: {count}")
        print(f"  - Overall throughput (E2E): {overall_throughput:.2f} samples/s")
        print(f"  - Effective throughput (workers only): {effective_throughput:.2f} samples/s")
        print(f"  - Average time per sample (E2E): {e2e_total_time / count:.2f}s")
        print(f"  - Average time per sample (workers only): {workers_time / count:.2f}s")

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total Processed: {count}")
    print(f"Score Method: {SCORE_METHOD}")
    if count > 0:
        method_label = "SubEM" if SCORE_METHOD == 'subem' else "EM"
        print(f"Average Accuracy ({method_label}): {total_score / count:.4f}")
    print(f"Merged results saved to: {FINAL_OUTPUT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    # 在 Linux 上使用 spawn 启动方式通常更安全，尤其是混合使用 cuda/npu 库时
    import multiprocessing

    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    args = parse_args()
    main(args)
