import os
import re
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

# Megatron side setting
MEGA_DP_SIZE = 4
MEGA_TP_SIZE = 4
MEGA_CP_SIZE = 2
# TP rank groups
MEGA_LD_RANKS = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
MEGA_LAYER_NAME = "model_0.module.module.output_layer_forward"
# Node IP of each tp rank group
MEGA_IPS = ["172.18.5.15", "172.18.5.15", "172.18.4.246", "172.18.4.246"]

# vLLM side setting
VLLM_DP_SIZE = 2
# only need to pick one rank from each tp group
VLLM_LD_RANKS = [[0], [4]]
VLLM_IPS = ["172.18.4.166", "172.18.4.166"]
VLLM_LAYER_NAME = "logits_processor_forward"
STEP = [3]
INTERACTION_TIME = 1
VOCAB_SIZE = 152064
K_MATCH_THRES = 0.3


def draw_train_infer_max_token_logp(vllm_probs, mega_probs, current_step, mega_id):
    m1 = vllm_probs.shape[0]
    m2 = mega_probs.shape[0]
    alpha = 0.1
    # === 选择两个明显不同的色系 ===
    colors_A = plt.cm.tab20c.colors[:4]
    colors_B = plt.cm.tab20c.colors[4:8]

    # 如果类别数 > 色系颜色数，可改用 tab20、viridis 等连续色系
    # 例如：colors_A = plt.cm.tab10(np.linspace(0, 1, m1))

    def lighten_color(color, amount=0.3):
        """
        将 RGB 颜色变亮（向白色靠近）
        color: (r, g, b, a) 或 (r, g, b)
        amount: 0~1，越大越亮
        """
        import matplotlib.colors as mc

        c = np.array(mc.to_rgb(color))
        lighter = c + (1 - c) * amount
        return np.clip(lighter, 0, 1)

    # === 绘图 ===
    plt.figure(figsize=(12, 7))

    # 绘制大类 A
    for i in range(m1):
        orig = vllm_probs[i]
        smoothed = pd.Series(orig).ewm(alpha=alpha).mean().values
        c = colors_A[i]
        c_light = lighten_color(c, amount=0.4)
        plt.plot(orig, color=c, linewidth=1.5, label=f'vLLM-{i + 1}')
        plt.plot(smoothed, color=c_light, linestyle='--', linewidth=1.5, label=f'vLLM-{i + 1} (smooth)')

    # 绘制大类 B
    for j in range(m2):
        orig = mega_probs[j]
        smoothed = pd.Series(orig).ewm(alpha=alpha).mean().values
        c = colors_B[j]
        c_light = lighten_color(c, amount=0.4)
        plt.plot(orig, color=c, linewidth=1.5, label=f'Mega-{j + 1}')
        plt.plot(smoothed, color=c_light, linestyle='--', linewidth=1.5, label=f'Mega-{j + 1} (smooth)')

    plt.title('Log p of vLLM and Mega')
    plt.xlabel('Token')
    plt.ylabel('log p')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放在图外避免遮挡
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'logp_step-{current_step}_traj-{mega_id}.png', dpi=300, bbox_inches='tight')
    plt.close()


def draw_train_infer_logits_diff(vllm_probs, mega_probs, current_step):
    diffs = np.abs(vllm_probs - mega_probs)
    corr = np.corrcoef(vllm_probs, mega_probs)[0, 1]

    # 绘图
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        vllm_probs,  # 横轴：推理概率
        mega_probs,  # 纵轴：训练概率
        c=diffs,  # 颜色映射依据
        cmap='viridis_r',  # 颜色映射方案（可选 'plasma', 'coolwarm' 等）
        alpha=0.7,
        edgecolors='none',
    )
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)

    plt.plot([0, 1], [0, 1], color='red', linestyle='--', linewidth=1, label='y = x')
    plt.legend()

    # 添加相关系数文本
    plt.text(
        0.75,
        0.05,
        f'Pearson r = {corr:.3f}',
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
    )

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_label('Absolute Difference (|Infer - Train|)', rotation=270, labelpad=20)

    # 设置坐标轴标签和标题
    plt.xlabel('Inference Probability')
    plt.ylabel('Training Probability')
    plt.title('Token-wise Probability Comparison')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 显示图形
    plt.tight_layout()
    plt.savefig(f'prob_difference_step-{current_step}.png', dpi=300, bbox_inches='tight')
    plt.close()
    return corr


def lcs(s1, s2):
    """
    s1: predicted tokens in training
    s2: predicted tokens in inference
    This is a non-classic LCS problem. Without tool-use, every token in s2 should be able to match continuously with
    tokens in s1. Even if several tokens do not match, it is still very possible that the next tokens in both sequences
    will match. With tool-use, it is possible for several consecutive tokens in s1 to fail to match with s2, but then
    starting from a certain token, the sequence can continue to match with s2 consecutively.
    """
    num_map = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    route_map = [[(0, 0) for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            if s1[i - 1] == s2[j - 1]:
                num_map[i][j] = num_map[i - 1][j - 1] + 1
                route_map[i][j] = (i - 1, j - 1)
            else:
                if num_map[i][j - 1] >= num_map[i][j]:
                    num_map[i][j] = num_map[i][j - 1]
                    route_map[i][j] = (i, j - 1)
                if num_map[i - 1][j] >= num_map[i][j]:
                    num_map[i][j] = num_map[i - 1][j]
                    route_map[i][j] = (i - 1, j)
                if num_map[i - 1][j - 1] >= num_map[i][j]:
                    num_map[i][j] = num_map[i - 1][j - 1]
                    route_map[i][j] = (i - 1, j - 1)

    i, j = len(s1), len(s2)
    index_record = [(i, j)]
    while i > 0 and j > 0:
        newi, newj = route_map[i][j]
        if index_record[-1][1] == newj:
            index_record[-1] = (newi, newj)
        else:
            index_record.append((newi, newj))
        i, j = newi, newj

    return num_map[-1][-1], index_record[:-1]


def lcs_match(mega_pred, candidate_reqs, merged_vllm_index_dict, left_traj):
    max_match_num = -1
    max_req_id = ""
    max_index = []

    for req_id in candidate_reqs:
        vllm_pred = merged_vllm_index_dict[req_id][..., 0]  # (seq,)
        match_num, index_record = lcs(mega_pred.cpu().numpy(), vllm_pred.cpu().numpy())

        if match_num > max_match_num:
            max_match_num = match_num
            max_req_id = req_id
            max_index = index_record
        elif match_num == max_match_num and req_id in left_traj:
            max_req_id = req_id
            max_index = index_record

    mega_index = []
    vllm_index = []
    for idx in max_index[::-1]:
        mega_index.append(idx[0] - 1)
        vllm_index.append(idx[1] - 1)
    mega_index = np.array(mega_index)
    vllm_index = np.array(vllm_index)
    return max_req_id, max_match_num, mega_index, vllm_index


def single_trajectory_match(divided_vllm_index, mega_pred):
    next_st = 0
    st_list = []
    current_match_num = 0
    vllm_total_len = 0
    for inter_id in range(INTERACTION_TIME):
        if divided_vllm_index[inter_id].nelement() == 0:
            break
        sub_max_match_num = -1
        st = next_st
        vllm_pred = divided_vllm_index[inter_id][..., 0]  # (sub_seq,)
        while st < mega_pred.shape[0]:
            valid_length = min(vllm_pred.shape[0], mega_pred.shape[0] - st)
            match_num = (vllm_pred[:valid_length] == mega_pred[st : st + valid_length]).int().sum().item()
            if match_num > sub_max_match_num:
                sub_max_match_num = match_num
                next_st = st
            st += 1
        current_match_num += sub_max_match_num
        st_list.append(next_st)
        next_st += vllm_pred.shape[0]
        vllm_total_len += vllm_pred.shape[0]
    return current_match_num, st_list, vllm_total_len


class MegaSegTree:
    def __init__(self, n):
        self.n = n
        self.has_one = [False] * (4 * n)  # 该区间是否包含至少一个 1
        self.lazy = [None] * (4 * n)  # None, 0, 或 1

    def _push(self, node, start, end):
        if self.lazy[node] is not None:
            color = self.lazy[node]
            self.has_one[node] = color == 1
            if start != end:
                self.lazy[node * 2] = color
                self.lazy[node * 2 + 1] = color
            self.lazy[node] = None

    def _update(self, node, start, end, l, r, color):
        self._push(node, start, end)
        if r < start or end < l:
            return
        if l <= start and end <= r:
            self.lazy[node] = color
            self._push(node, start, end)
            return
        mid = (start + end) // 2
        self._update(node * 2, start, mid, l, r, color)
        self._update(node * 2 + 1, mid + 1, end, l, r, color)
        # 关键：合并子区间 —— 只要任一子区间有 1，当前区间就有 1
        # 但必须先 push 子节点以获取最新值
        self._push(node * 2, start, mid)
        self._push(node * 2 + 1, mid + 1, end)
        self.has_one[node] = self.has_one[node * 2] or self.has_one[node * 2 + 1]

    def update(self, l, r, color):
        """将 [l, r] 染成 color (0 或 1)"""
        self._update(1, 0, self.n - 1, l, r, color)

    def _query(self, node, start, end, l, r):
        if r < start or end < l:
            return False
        self._push(node, start, end)
        if l <= start and end <= r:
            return self.has_one[node]
        mid = (start + end) // 2
        left_has = self._query(node * 2, start, mid, l, r)
        right_has = self._query(node * 2 + 1, mid + 1, end, l, r)
        return left_has or right_has

    def query_has_one(self, l, r):
        """返回 [l, r] 中是否存在 1"""
        return self._query(1, 0, self.n - 1, l, r)


def greedy_match(edges, divided_vllm_index_dict, merged_mega_index_list):
    matched_traj = set()
    # matched_idx = set()
    matched_idx = [MegaSegTree(merged_mega_index_list[i].shape[0]) for i in range(len(merged_mega_index_list))]
    match_map = []

    for w, traj_id, idx, match_num, st_list in edges:
        for i in range(len(st_list)):
            mega_index = []
            vllm_index = []
            vllm_st = 0
            mega_pred = merged_mega_index_list[idx][..., 0]
            # if traj_id not in matched_traj and idx not in matched_idx:
            vllm_sub_seq = divided_vllm_index_dict[traj_id][i].shape[0]

            if traj_id not in matched_traj and not matched_idx[idx].query_has_one(
                st_list[i], min(st_list[i] + vllm_sub_seq, mega_pred.shape[0])
            ):
                new_mega_index = list(range(st_list[i], min(st_list[i] + vllm_sub_seq, mega_pred.shape[0])))
                mega_index.extend(new_mega_index)
                new_vllm_index = list(range(vllm_st, vllm_st + len(new_mega_index)))
                vllm_index.extend(new_vllm_index)
                vllm_st += len(new_vllm_index)
                mega_index = np.array(mega_index)
                vllm_index = np.array(vllm_index)

                match_map.append(
                    {
                        "mega_id": idx,
                        "req_id": traj_id,
                        "match_num": match_num,
                        "mega_index": mega_index,
                        "vllm_index": vllm_index,
                    }
                )

                matched_traj.add(traj_id)
                matched_idx[idx].update(st_list[i], min(st_list[i] + vllm_sub_seq, mega_pred.shape[0]) - 1, 1)

        if len(match_map) == len(divided_vllm_index_dict.keys()):
            break
    return match_map


def k_gram_match(mega_index_list, vllm_index_dict, k=5, base=256, mod=2**61 - 1):
    """Quickly match megatron prediction and vllm prediction, provide approximated match map."""

    def hash_list(arr):
        h = 0
        for i in range(len(arr)):
            h = (h * base + arr[i]) % mod
        return h

    inverted_index = defaultdict(set)
    power = pow(base, k - 1, mod)

    for idx, mega_index in enumerate(mega_index_list):
        mega_pred = mega_index[..., 0].squeeze().cpu().numpy()  # (seq,)
        h = hash_list(mega_pred[:k])
        inverted_index[h].add(idx)
        for i in range(1, mega_index.shape[0] - k + 1):
            h = (h - mega_pred[i - 1] * power) % mod
            h = (h * base + mega_pred[i + k - 1]) % mod
            inverted_index[h].add(idx)

    # candidate_map = [[] for _ in range(len(mega_index_list))]
    candidate_map = {}
    left_idx = list(range(len(mega_index_list)))
    for traj_id, vllm_index in vllm_index_dict.items():
        vote_count = defaultdict(int)
        candidate_map[traj_id] = []
        vllm_pred = vllm_index[..., 0].cpu().numpy()  # (seq,)
        h = hash_list(vllm_pred[:k])
        for idx in inverted_index[h]:
            vote_count[idx] += 1
        for i in range(1, vllm_index.shape[0] - k + 1):
            h = (h - vllm_pred[i - 1] * power) % mod
            h = (h * base + vllm_pred[i + k - 1]) % mod
            for idx in inverted_index[h]:
                vote_count[idx] += 1

        sorted_count = sorted(vote_count.items(), key=lambda x: x[1], reverse=True)
        for idx, v in sorted_count:
            if v / (vllm_index.shape[0] - k + 1) > K_MATCH_THRES:
                candidate_map[traj_id].append(idx)
                if idx in left_idx:
                    left_idx.remove(idx)

    return candidate_map


def find_inference_step_range(train_step, last_train_step=-1):
    train_mtime = 0
    for tp_rank in MEGA_LD_RANKS:
        for rank in tp_rank:
            train_file_path = f"./mathtool_actor_sentinel_dump_data/step_{train_step}/rank_{rank}/tensor_data.pkl"
            train_mtime = max(train_mtime, os.path.getmtime(train_file_path))
    last_train_mtime = 0
    if last_train_step != -1:
        for tp_rank in MEGA_LD_RANKS:
            for rank in tp_rank:
                train_file_path = (
                    f"./mathtool_actor_sentinel_dump_data/step_{last_train_step}/rank_{rank}/tensor_data.pkl"
                )
                last_train_mtime = max(last_train_mtime, os.path.getmtime(train_file_path))
    else:
        last_train_mtime = train_mtime - 3600 * 4

    infer_step_folders = "./mathtool_rollout_sentinel_dump_data"
    inference_steps = []
    for item in os.listdir(infer_step_folders):
        item_path = os.path.join(infer_step_folders, item)
        if os.path.isdir(item_path) and "step" in item:
            vllm_step = int(re.search("step_(\d+)", item)[1])
            for tp_rank in VLLM_LD_RANKS:
                for rank in tp_rank:
                    infer_file_path = os.path.join(item_path, f"rank_{rank}", "tensor_data.pkl")
                    if os.path.exists(infer_file_path):
                        infer_mtime = os.path.getmtime(infer_file_path)

                        if (
                            infer_mtime < train_mtime
                            and infer_mtime > last_train_mtime
                            and vllm_step not in inference_steps
                        ):
                            inference_steps.append(vllm_step)
    inference_steps = sorted(inference_steps)
    return inference_steps


def load_mega_logits(step):
    # Load training logits
    merged_mega_logits_list = []
    merged_mega_index_list = []
    merged_mega_prob_list = []
    divided_mega_logits_list = []
    divided_mega_index_list = []
    divided_mega_prob_list = []
    partial_vocab = VOCAB_SIZE // MEGA_TP_SIZE
    for tp_range, node_ip in zip(MEGA_LD_RANKS, MEGA_IPS):
        tmp_traj_num = -1
        mega_logits_list = []
        for rank in tp_range:
            mega_path = f"./mathtool_actor_sentinel_dump_data/step_{step}/{node_ip}-rank_{rank}/tensor_data.pkl"
            mega_logits = torch.load(mega_path)  # "topk_logits": [(2,seq,mb,topk) * traj]
            if tmp_traj_num == -1:
                tmp_traj_num = len(mega_logits["top100_logits"][MEGA_LAYER_NAME])
            else:
                expected_len = len(mega_logits["top100_logits"][MEGA_LAYER_NAME])
                if tmp_traj_num != expected_len:
                    raise ValueError(
                        f"Data length mismatch: tmp_traj_num ({tmp_traj_num}) does not match "
                        f"expected length ({expected_len})"
                    )
            mega_logits_list.append(mega_logits)  # [[(2,seq,mb,topk) * traj] * tp]

        tmp_mega_logits_list = []
        tmp_mega_index_list = []
        tmp_mega_prob_list = []
        # Process training logits to get final logits, index, and prob
        for i in range(tmp_traj_num):
            logits_list = [x["top100_logits"][MEGA_LAYER_NAME][i].npu() for x in mega_logits_list]
            mb = logits_list[0].shape[2]
            for j in range(mb):
                ori_logits_list = []
                ori_index_list = []
                for k, logit in enumerate(logits_list):  # (2,seq,mb,topk)
                    ori_logit = logit[0, :, j, :]  # (seq,topk)
                    index = logit[1, :, j, :].long()  # (seq,topk)
                    index += k * partial_vocab  # (seq,topk)
                    ori_logits_list.append(ori_logit)  # [(seq,topk) * tp]
                    ori_index_list.append(index)  # [(seq,topk) * tp]
                mega_logits = torch.cat(ori_logits_list, dim=-1)  # (seq,topk*tp)
                mega_index = torch.cat(ori_index_list, dim=-1)  # (seq,topk*tp)
                top_mega_logits, top_index = mega_logits.sort(descending=True)
                top_mega_logits = top_mega_logits[..., :100]  # (seq,topk)
                mega_index = torch.gather(mega_index, dim=-1, index=top_index[..., :100])  # (seq,topk)

                tmp_mega_logits_list.append(top_mega_logits)  # [(seq,topk) * (mb * tmp_traj)]
                tmp_mega_index_list.append(mega_index)  # [(seq,topk) * (mb * tmp_traj)]

                safe_exp_logits = torch.exp(
                    top_mega_logits - torch.amax(top_mega_logits, dim=-1, keepdim=True)
                )  # (seq,topk)
                tail_exp = (VOCAB_SIZE - 100) * safe_exp_logits[..., -1]  # (seq,)
                prob = safe_exp_logits / (torch.sum(safe_exp_logits, dim=-1) + tail_exp).unsqueeze(-1)  # (seq,topk)
                tmp_mega_prob_list.append(prob)  # [(seq,topk) * (mb * tmp_traj)]
        divided_mega_logits_list.append(tmp_mega_logits_list)  # [[(seq,topk) * (mb * tmp_traj)] * cp]
        divided_mega_index_list.append(tmp_mega_index_list)  # [[(seq,topk) * (mb * tmp_traj)] * cp]
        divided_mega_prob_list.append(tmp_mega_prob_list)  # [[(seq,topk) * (mb * tmp_traj)] * cp]
        if len(divided_mega_logits_list) == MEGA_CP_SIZE:
            cp_traj_num = len(divided_mega_logits_list[0])
            for j in range(cp_traj_num):
                logits = [x[j] for x in divided_mega_logits_list]
                index = [x[j] for x in divided_mega_index_list]
                prob = [x[j] for x in divided_mega_prob_list]
                merged_mega_logits_list.append(torch.cat(logits, dim=0))
                merged_mega_index_list.append(torch.cat(index, dim=0))
                merged_mega_prob_list.append(torch.cat(prob, dim=0))
            divided_mega_logits_list = []
            divided_mega_index_list = []
            divided_mega_prob_list = []

    traj_num = len(merged_mega_logits_list)
    print("Mega trajectory number:", traj_num)
    return merged_mega_logits_list, merged_mega_index_list, merged_mega_prob_list


def load_vllm_logits(step):
    # Load inference logits
    vllm_logits_dict = {}
    # 混合异步下，会有上一个 step 生成的traj在当前step被送进actor，所以多往前找一个step
    if step > 0:
        inference_steps = [step - 1, step]
    else:
        inference_steps = [step]

    for tp_range, node_ip in zip(VLLM_LD_RANKS, VLLM_IPS):
        rank = tp_range[0]  # vLLM will gather all logits in the logits processor, so all tp share a same logits data
        for i in inference_steps:
            vllm_path = f"./mathtool_rollout_sentinel_dump_data/step_{i}/{node_ip}-rank_{rank}/tensor_data.pkl"
            if not os.path.exists(vllm_path):
                continue
            vllm_logits = torch.load(vllm_path)  # "topk_logits": [(2,req,topk) * req_num]
            if "input_id" not in vllm_logits.keys():
                raise RuntimeError("Not found vLLM req_id information")

            # Double check train step
            for l in range(len(vllm_logits["input_id"][VLLM_LAYER_NAME])):
                if len(vllm_logits["input_id"][VLLM_LAYER_NAME][l]) > 0:
                    req_id = vllm_logits["input_id"][VLLM_LAYER_NAME][l][0]
                    train_step = int(re.search(".*-(trainstep)-(\d+)", req_id)[2])
                    if 0 <= train_step - step < -2:
                        raise RuntimeError(
                            f"Train step in {vllm_path} and target step are mismatched. "
                            f"Got {train_step} and target is {step}"
                        )

                    for req_id in vllm_logits["input_id"][VLLM_LAYER_NAME][l]:
                        if req_id not in vllm_logits_dict.keys():
                            vllm_logits_dict[req_id] = []

                    len_input_id = len(vllm_logits["input_id"][VLLM_LAYER_NAME][l])
                    len_top100 = vllm_logits["top100_logits"][VLLM_LAYER_NAME][l].shape[1]
                    if len_input_id != len_top100:
                        raise ValueError(
                            f"Dimension mismatch at layer {l}: input_id length ({len_input_id}) "
                            f"does not match top100_logits shape ({len_top100})"
                        )
                    for j in range(len(vllm_logits["input_id"][VLLM_LAYER_NAME][l])):
                        req_id = vllm_logits["input_id"][VLLM_LAYER_NAME][l][j]
                        logits = vllm_logits["top100_logits"][VLLM_LAYER_NAME][l][:, j, :]  # （2,topk）
                        vllm_logits_dict[req_id].append(logits.unsqueeze(1).npu())  # ([2,1,topk] * sub_seq)

    for req_id in vllm_logits_dict.keys():
        vllm_logits_dict[req_id] = torch.cat(vllm_logits_dict[req_id], dim=1)  # (2,sub_seq,topk)

    merged_vllm_logits_dict = {}
    merged_vllm_index_dict = {}
    merged_vllm_prob_dict = {}
    divided_vllm_logits_dict = {}
    divided_vllm_index_dict = {}
    inter_time = 0
    # Merge output of different interactions from the same trajectory
    for req_id, v in vllm_logits_dict.items():  # (2,sub_seq,topk)
        req_id_seg = req_id.split('-')
        traj_id = '-'.join(req_id_seg[:-2])
        divided_vllm_logits_dict[traj_id] = [v]  # (2,sub_seq,topk)
        divided_vllm_index_dict[traj_id] = [v]  # (2,sub_seq,topk)

    for traj_id in divided_vllm_logits_dict.keys():
        merged_vllm_logits_dict[traj_id] = torch.cat(divided_vllm_logits_dict[traj_id], dim=1)  # (2,seq,topk)
        merged_vllm_index_dict[traj_id] = merged_vllm_logits_dict[traj_id][1].long()  # (seq,topk)
        merged_vllm_logits_dict[traj_id] = merged_vllm_logits_dict[traj_id][0]  # (seq,topk)

        for i in range(len(divided_vllm_logits_dict[traj_id])):
            if divided_vllm_logits_dict[traj_id][i].nelement() > 0:
                divided_vllm_index_dict[traj_id][i] = divided_vllm_logits_dict[traj_id][i][1].long()  # (sub_seq,topk)
                divided_vllm_logits_dict[traj_id][i] = divided_vllm_logits_dict[traj_id][i][0]

        logit = merged_vllm_logits_dict[traj_id]  # (seq,topk)
        safe_exp_logits = torch.exp(logit - torch.amax(logit, dim=-1, keepdim=True))  # (seq,topk)
        tail_exp = (VOCAB_SIZE - 100) * safe_exp_logits[..., -1]  # (seq,)
        prob = safe_exp_logits / (torch.sum(safe_exp_logits, dim=-1) + tail_exp).unsqueeze(-1)  # (seq,topk)
        merged_vllm_prob_dict[traj_id] = prob  # (seq,topk)

    print("vllm trajectory number:", len(merged_vllm_logits_dict.keys()))
    return (
        merged_vllm_logits_dict,
        merged_vllm_index_dict,
        merged_vllm_prob_dict,
        divided_vllm_logits_dict,
        divided_vllm_index_dict,
    )


@torch.no_grad()
def top_logits_match(step):
    # Load training logits
    merged_mega_logits_list, merged_mega_index_list, merged_mega_prob_list = load_mega_logits(step)

    # Load inference logits
    (
        merged_vllm_logits_dict,
        merged_vllm_index_dict,
        merged_vllm_prob_dict,
        divided_vllm_logits_dict,
        divided_vllm_index_dict,
    ) = load_vllm_logits(step)

    if os.path.exists(f"./outputs/rollout_actor_match_step_{step}.pkl"):
        with open(f"./outputs/rollout_actor_match_step_{step}.pkl", "rb") as f:
            match_map = pickle.load(f)
    else:
        candidate_map = k_gram_match(merged_mega_index_list, merged_vllm_index_dict)

        edges = []

        for traj_id in candidate_map:
            for idx in candidate_map[traj_id]:
                current_match_num, st_list, vllm_total_length = single_trajectory_match(
                    divided_vllm_index_dict[traj_id], merged_mega_index_list[idx][..., 0].squeeze()
                )
                edges.append((float(current_match_num) / vllm_total_length, traj_id, idx, current_match_num, st_list))
        edges.sort(key=lambda x: x[0], reverse=True)

        match_map = greedy_match(edges, divided_vllm_index_dict, merged_mega_prob_list)
        # Save match map for future use
        with open(f"./outputs/rollout_actor_match_step_{step}.pkl", 'wb') as f:
            pickle.dump(match_map, f)

    traj_num = len(match_map)
    match_ratio = 0.0
    target_draw_traj_id = []
    for i in range(len(match_map)):
        print(
            match_map[i]['mega_id'],
            match_map[i]['req_id'],
            match_map[i]['match_num'],
            match_map[i]['match_num'] / len(match_map[i]['vllm_index']),
            match_map[i]["mega_index"][0],
            match_map[i]["mega_index"][-1],
        )
        match_ratio += match_map[i]['match_num'] / len(match_map[i]['vllm_index'])
        if match_map[i]['mega_id'] not in target_draw_traj_id:
            target_draw_traj_id.append(match_map[i]['mega_id'])
    match_ratio /= len(match_map)
    print(len(match_map), len(target_draw_traj_id), f"Matched idx {target_draw_traj_id}")
    print(f"Average match ratio: {match_ratio}")

    vllm_max_token_probs = [[] for _ in range(traj_num)]
    mega_max_token_probs = [[] for _ in range(traj_num)]
    vllm_probs = []
    mega_probs = []
    all_kl = []
    complete_mega_probs = [np.full(merged_mega_prob_list[i].shape[0], -np.inf) for i in target_draw_traj_id]
    complete_vllm_probs = [np.full(merged_mega_prob_list[i].shape[0], -np.inf) for i in target_draw_traj_id]
    for i in range(traj_num):
        mega_id = match_map[i]["mega_id"]  # int
        req_id = match_map[i]["req_id"]  # str
        mega_index = match_map[i]['mega_index']  # (seq,)
        vllm_index = match_map[i]['vllm_index']  # (seq,)

        mega_p = merged_mega_prob_list[mega_id][mega_index].cpu()  # (seq, topk)
        vllm_p = merged_vllm_prob_dict[req_id][vllm_index].cpu()  # (seq, topk)
        mega_pred = merged_mega_index_list[mega_id][mega_index].cpu()  # (seq, topk)
        vllm_pred = merged_vllm_index_dict[req_id][vllm_index].cpu()  # (seq, topk)

        mega_prob_tensor = torch.zeros((vllm_p.shape[0], VOCAB_SIZE))  # (seq, vocab)
        vllm_prob_tensor = torch.zeros((vllm_p.shape[0], VOCAB_SIZE))  # (seq, vocab)

        mega_prob_tensor[torch.arange(mega_prob_tensor.shape[0]).view(-1, 1), mega_pred] = mega_p  # (seq, vocab)
        vllm_prob_tensor[torch.arange(vllm_prob_tensor.shape[0]).view(-1, 1), vllm_pred] = vllm_p  # (seq, vocab)

        valid_prob_index = (mega_prob_tensor > 1e-2) & (vllm_prob_tensor > 1e-2)
        mega_probs.extend(mega_prob_tensor[valid_prob_index].cpu().tolist())
        vllm_probs.extend(vllm_prob_tensor[valid_prob_index].cpu().tolist())

        mega_max_probs = mega_prob_tensor[
            torch.arange(mega_prob_tensor.shape[0]).view(-1, 1), vllm_pred[:, 0].view(-1, 1)
        ].squeeze()  # 以vllm预测的tensor为准
        vllm_max_probs = vllm_prob_tensor[
            torch.arange(vllm_prob_tensor.shape[0]).view(-1, 1), vllm_pred[:, 0].view(-1, 1)
        ].squeeze()

        if mega_id in target_draw_traj_id:
            idx = target_draw_traj_id.index(mega_id)
            mega_prob = np.log(mega_max_probs.numpy())
            vllm_prob = np.log(vllm_max_probs.numpy())

            complete_mega_probs[idx][mega_index] = mega_prob
            complete_vllm_probs[idx][mega_index] = vllm_prob

        mega_prob_tensor = mega_prob_tensor.clamp(min=1e-20)
        vllm_prob_tensor = vllm_prob_tensor.clamp(min=1e-20)
        kl = (vllm_prob_tensor * (torch.log(vllm_prob_tensor) - torch.log(mega_prob_tensor))).sum(dim=1)  # (seq,)
        all_kl.append(kl)

    for i in target_draw_traj_id:
        idx = target_draw_traj_id.index(i)
        draw_train_infer_max_token_logp(
            np.expand_dims(complete_vllm_probs[idx], 0), np.expand_dims(complete_mega_probs[idx], 0), current_step, i
        )
    all_kl = torch.cat(all_kl[:], dim=0)  # (seq * traj,)
    avg_kl_div = all_kl.mean().item()  # token level kl-divergence
    print(f"Average KL divergence: {avg_kl_div}")

    vllm_probs = np.array(vllm_probs)
    mega_probs = np.array(mega_probs)

    # Draw figures
    corr = draw_train_infer_logits_diff(vllm_probs, mega_probs, current_step)
    print(f"Pearson correlation: {corr}")


def router_index_compare():
    mega_path = "./sentinel_dump_data_router_index/step_0/rank_0/tensor_data.pkl"
    vllm_path = "./vllm_sentinel_dump_data_router_index/step_0/rank_0/tensor_data.pkl"

    mega_index = torch.load(mega_path)['raw_data']
    vllm_logits = torch.load(vllm_path)['raw_data']

    layers = 48
    mega_pre_id = 5
    k = 8

    for i in range(layers):
        mega_layer_index = mega_index[f'module.decoder.layers.{i}.mlp.router_forward'][0][:mega_pre_id]
        vllm_layer_logits = vllm_logits[f'model.layers.{i}.mlp.gate_forward'][0]
        vllm_layer_index = vllm_layer_logits.sort(descending=True, dim=-1)[1][:, :8]

        print(i, mega_layer_index, vllm_layer_index)


if __name__ == '__main__':
    for current_step in STEP:
        top_logits_match(current_step)
