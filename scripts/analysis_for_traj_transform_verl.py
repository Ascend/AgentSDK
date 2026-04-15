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

import json
import os
import re
from json import JSONDecoder
from collections import defaultdict

ROLLOUT_FILE_PATH = "/opt/DPC/models/z00943413/AgenticRL_5.0/AgenticRL_verl/origin_rollout/rollout_trajectories_20260206_091205.json"

VERL_DIR = "/opt/DPC/models/z00943413/AgenticRL_5.0/AgenticRL_verl/rollout_path_0206_step_10/"

CHECK_ITERATIONS = 10
EXPECTED_COUNT = 8


class TextNormalizer:
    @staticmethod
    def clean(text):
        """
        Normalize text for comparison:
        1. Convert to string
        2. Remove special tokens (<|im_end|>, <|im_start|>)
        3. [Key change] Remove <think> tags, since Verl output typically does not contain chain-of-thought tags
        4. Collapse whitespace characters
        """
        if text is None:
            return ""
        text = str(text)

        text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
        text = text.replace("<think>", "").replace("</think>", "")

        return " ".join(text.split())


class TrajectoryAligner:
    """
    Responsible for reconstructing Rollout structured data into Verl string format
    """

    @staticmethod
    def reconstruct_verl_string(chat_completions):
        """
        Core logic: reconstruct chat_completions list into Verl input and output strings
        """
        temp_msgs = chat_completions.copy()

        system_content = ""
        user_content = ""

        if temp_msgs and temp_msgs[0]["role"] == "system":
            system_content = temp_msgs.pop(0)["content"]
        
        if temp_msgs and temp_msgs[0]["role"] == "user":
            user_content = temp_msgs.pop(0)["content"]
        
        input_str = f"system\n{system_content}\n\nuser\n{user_content}\nassistant\n"

        output_str = ""

        for msg in temp_msgs:
            role = msg.get('role')
            content = msg.get('content', '')

            if role == "assistant":
                output_str += content
            elif role == 'tool':
                tool_part = f"\nuser\n<tool_response>\n{content}\n</tool_response>\nassistant\n"
                output_str += tool_part
            else:
                output_str += content
        
        return input_str, output_str
    

class ConsistencyChecker:
    def __init__(self):
        pass
    
    def _parse_stacked_json(self, content):
        """
        Parse stacked JSON strings (handle non-standard JSONL)
        """
        decoder = JSONDecoder()
        content = content.strip()
        objs = []
        while content:
            try:
                content = content.lstrip()
                if not content: break
                obj, idx = decoder.raw_decode(content)
                objs.append(obj)
                content = content[idx:]
            except json.JSONDecodeError:
                print("Warning: Stopped parsing rollout file due to garbage at end.")
                break
        return objs

    def load_rollout_data(self, path):
        print(f"Loading Rollout: {path}")
        if not os.path.exists(path):
            print(f"[ERROR] file not found: {path}")
            return {}
        
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        data_list = self._parse_stacked_json(content)

        iter_map = {}
        for item in data_list:
            it = str(item.get("iteration"))
            trajs = item.get("trajectories", [])

            if it in iter_map:
                iter_map[it].extend(trajs)
            else:
                iter_map[it] = trajs

        print(f" - Loaded {len(iter_map)} iterations.")
        return iter_map
    
    def load_verl_data(self, iter_num):
        """
        Load Verl data
        """
        path = os.path.join(VERL_DIR, f"{iter_num}.jsonl")
        if not os.path.exists(path):
            return {}
    
        data_map = defaultdict(set)

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    in_key = TextNormalizer.clean(item.get("input", ""))
                    out_key = TextNormalizer.clean(item.get("output", ""))
                    
                    data_map[in_key].add(out_key)
                except:
                    pass
        total_samples = sum(len(outputs) for outputs in data_map.values())
        if total_samples != EXPECTED_COUNT:
            print(f"[WARNING] Sample count mismatch! Expected {EXPECTED_COUNT}, but got {total_samples}")

        return data_map
    
    def run(self):
        r_map = self.load_rollout_data(ROLLOUT_FILE_PATH)
        if not r_map: return
        
        print("=" * 60)

        for i in range(CHECK_ITERATIONS):
            r_key = str(i)
            v_num = i + 1

            print(f"\n>>> Checking Iteration {i} (Verl file: {v_num}.jsonl)")

            r_list = r_map.get(r_key, [])
            v_map = self.load_verl_data(v_num)

            if not r_list:
                print(f" [SKIP] No rollout data for iter {i}")
                continue
            
            if not v_map:
                print(f" [SKIP] No verl data for iter {i}")
                continue

            match_count = 0
            mismatch_count = 0
            missing_count = 0

            for idx, r_item in enumerate(r_list):
                r_in_str, r_out_str = TrajectoryAligner.reconstruct_verl_string(r_item["chat_completions"])

                r_in_clean = TextNormalizer.clean(r_in_str)
                r_out_clean = TextNormalizer.clean(r_out_str)

                if r_in_clean in v_map:
                    existing_outputs = v_map[r_in_clean]

                    if r_out_clean in existing_outputs:
                        match_count += 1
                    else:
                        mismatch_count += 1
                        print(f"\n [MISMATCH] Input found, but Output mismatch. Rollout Index {idx}")
                        v_out_sample = next(iter(existing_outputs))
                        self.print_diff(r_out_clean, v_out_sample)
                else:
                    missing_count += 1
                    print(f"\n [MISSING] Rollout input NOT found in Verl file. Index {idx}")
            
            total = len(r_list)
            print(f" Result: {total} items | OK: {match_count} | Diff: {mismatch_count} | Missing: {missing_count}")
            if match_count == total and total > 0:
                print(f" \033[92m[SUCCESS] Iteration {i} is perfectly aligned. \033[0m")
        
    def print_diff(self, str1, str2):
        """
        Compare and print difference points
        """
        limit = min(len(str1), len(str2))
        diff_idx = 0
        while diff_idx < limit and str1[diff_idx] == str2[diff_idx]:
            diff_idx += 1
        
        print(f"Diff starts at char {diff_idx}")
        start = max(0, diff_idx - 30)
        end = min(limit, diff_idx + 80)
        print(f"Rollout (clean): ...{str1[start:end]}...")
        print(f"Verl (clean): ...{str2[start:end]}...")


if __name__ == "__main__":
    checker = ConsistencyChecker()
    checker.run()
