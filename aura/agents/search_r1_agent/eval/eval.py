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
import torch
import random
from datasets import load_dataset
import requests
import re
import string
import json
from tqdm import tqdm
import os
from collections.abc import Mapping, Sequence
from pathlib import Path


# ================= Reward / Scoring 代码 =================
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
    print("------------------------------")
    print(prediction, golden_answers)
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
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) < 1:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    print(f"answer is {solution_str} and ground_truth is {ground_truth}")
    answer = extract_solution(solution_str=solution_str)
    do_print = True

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

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


# ===================辅助函数，直接从parquet进行读取数据================
def read_trivia_qa_validation(data_path):
    dataset = load_dataset("parquet", data_files={"validation": f"{data_path}/validation-*.parquet"})

    val_ds = dataset["validation"]
    print(val_ds[0])
    return val_ds.to_list()


def read_trivia_qa_train(data_path):
    dataset = load_dataset("parquet", data_files={"validation": f"{data_path}/train.parquet"})

    val_ds = dataset["validation"]
    print(val_ds[0])
    return val_ds.to_list()


def read_nq_search_test(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        # 如果文件是列表格式 [ {...}, {...} ]
        try:
            dataset_raw = json.load(f)
        except json.JSONDecodeError:
            # 如果文件是 jsonl 格式 (每行一个 json)
            f.seek(0)
            dataset_raw = [json.loads(line) for line in f]
    return dataset_raw


# ================= 2. 推理与搜索基础架构 =================
# Define the custom stopping criterion
class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        # Encode the string so we have the exact token-IDs pattern
        self.target_ids = [
            tokenizer.encode(target_sequence, add_special_tokens=False) for target_sequence in target_sequences
        ]
        self.target_lengths = [len(target_id) for target_id in self.target_ids]
        self._tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Make sure the target IDs are on the same device
        targets = [torch.as_tensor(target_id, device=input_ids.device) for target_id in self.target_ids]

        if input_ids.shape[1] < min(self.target_lengths):
            return False

        # Compare the tail of input_ids with our target_ids
        for i, target in enumerate(targets):
            if torch.equal(input_ids[0, -self.target_lengths[i] :], target):
                return True

        return False


def get_query(text):
    import re

    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    if matches:
        return matches[-1]
    else:
        return None


def search(query: str):
    payload = {"queries": [query], "topk": 3, "return_scores": True}
    results = requests.post("http://7.246.80.27:8000/retrieve", json=payload).json()['result']

    def _passages2string(retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx + 1}(Title: {title}) {text}\n"
        return format_reference

    return _passages2string(results[0])


# ================= 3. 主评测流程 =================
def eval(question, ground_truth, tokenizer, model):
    device = "npu"
    question = question.strip()
    # Initialize the stopping criteria
    if question[-1] != '?':
        question += '?'
    curr_eos = [151645, 151643]  # for Qwen2.5 series models
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'

    # Prepare the message
    prompt = f"""Answer the given question. \
        You must conduct reasoning inside <think> and </think> first every time you get new information. \
        After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
        You can search as many times as your want. \
        If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"""

    target_sequences = ["</search>", " </search>", "</search>\n", " </search>\n", "</search>\n\n", " </search>\n\n"]
    stopping_criteria = transformers.StoppingCriteriaList([StopOnSequence(target_sequences, tokenizer)])

    cnt = 0

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
        )

    print('\n\n################# [Start Reasoning + Searching] ##################\n\n')
    # print(prompt)
    # Encode the chat-formatted prompt and move it to the correct device
    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        attention_mask = torch.ones_like(input_ids)

        # Generate text with the stopping criteria
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
        )

        if outputs[0][-1].item() in curr_eos:
            generated_tokens = outputs[0][input_ids.shape[1] :]
            output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # print(f"answer is {output_text} and ground_truth is {ground_truth}")  # 答案 reward，
            reward = compute_score_subem(output_text, ground_truth)
            return reward, prompt + output_text

        generated_tokens = outputs[0][input_ids.shape[1] :]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        tmp_query = get_query(tokenizer.decode(outputs[0], skip_special_tokens=True))
        if tmp_query:
            # print(f'searching "{tmp_query}"...')
            search_results = search(tmp_query)
        else:
            search_results = ''

        search_text = curr_search_template.format(output_text=output_text, search_results=search_results)
        prompt += search_text
        cnt += 1
        # print(search_text)


def main():
    # 配置
    DATA_FILE = "/opt/DPC/models/dataset/search-r1/aura/ori/nq_hotpotqa/"  # json 文件路径
    OUTPUT_FILE = "tmp_test0105.jsonl"
    model_id = "/opt/DPC/models/AgenticRL_3.0/AgenticRL_math_varify/Search-R1-main/ckpt/qwen25_7B_140_iter/mg2hf"
    print(f"Loading data from {DATA_FILE}...")
    p = Path(DATA_FILE)
    if p.is_file():
        dataset_raw = read_nq_search_test(DATA_FILE)
    elif p.is_dir():
        dataset_raw = read_trivia_qa_train(DATA_FILE)
    else:
        raise ValueError("Unsupported file type:")
    results = []
    total_score = 0
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)
    dataset = dataset_raw[:100]
    # Initialize the tokenizer and model
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=None,  # 禁用自动映射
    ).to("npu")
    model.eval()
    print(f"Starting evaluation on {len(dataset)} samples...")

    f_out = open(OUTPUT_FILE, 'a', encoding='utf-8')

    try:
        for idx, item in tqdm(enumerate(dataset), total=len(dataset)):
            try:
                question = item['question']
                raw_ground_truth = item['reward_model']['ground_truth']
                # question = item['question']
                # raw_ground_truth = item['answer']['normalized_aliases']

                if isinstance(raw_ground_truth, Mapping):
                    # 已经是 dict，假定格式正确
                    ground_truth_dict = raw_ground_truth

                elif isinstance(raw_ground_truth, str):
                    ground_truth_dict = {"target": [raw_ground_truth]}

                elif isinstance(raw_ground_truth, Sequence):
                    # list / tuple 等，但排除 str（上面已处理）
                    print("here")
                    ground_truth_dict = {"target": list(raw_ground_truth)}

                else:
                    raise TypeError(f"Unsupported ground_truth type: {type(raw_ground_truth)}")

                reward, output_text = eval(question, ground_truth_dict, tokenizer, model)

            except Exception as e:
                print(f"[Skip] idx={idx}, error={e}")
                continue

            total_score += reward
            results.append(reward)

            result_item = {
                "index": idx,
                "question": question,
                "ground_truth": raw_ground_truth,
                "prediction": output_text,
                "score": reward,
            }

            # ===== 实时写出 =====
            f_out.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            f_out.flush()

    finally:
        f_out.close()

    print("\n" + "=" * 30)
    print("Evaluation Finished.")
    print(f"Total Samples: {len(dataset)}")
    print(f"Average Accuracy (EM): {total_score / len(dataset):.4f}")
    print(f"Details saved to: {OUTPUT_FILE}")
    print("=" * 30)


if __name__ == "__main__":
    main()
    # print(em_check("Wilhelm Conrad Röntgen",['Wilhelm Conrad Röntgen']))
