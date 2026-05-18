_SOFTMAX_INDICES_CACHE_LRU = {}
_ACCUMULATE_LIST_CACHE_LRU = {}


def accumulate_list(input_list):
    """
    缓存优化版本 - 使用LRU缓存机制，最多缓存一个条目以避免重复计算
    """
    # 创建缓存键（包含设备信息以确保正确性）
    cache_key = (str(input_list), str(prev_attn_out.device))

    # 检查缓存命中
    if cache_key in _ACCUMULATE_LIST_CACHE_LRU:
        return _ACCUMULATE_LIST_CACHE_LRU[cache_key]

    # 未命中缓存，执行正常计算
    if not input_list:
        result = torch.tensor([0], dtype=torch.int64, device=prev_attn_out.device)
    else:
        input_tensor = torch.tensor(input_list, dtype=torch.int64, device=prev_attn_out.device)
        cumsum_result = torch.cumsum(input_tensor, dim=0)
        result = torch.cat([torch.tensor([0], dtype=torch.int64, device=prev_attn_out.device), cumsum_result])

    # 维护缓存大小为1，只留最新的条目
    if len(_ACCUMULATE_LIST_CACHE_LRU) >= 1:
        old_key = next(iter(_ACCUMULATE_LIST_CACHE_LRU))
        del _ACCUMULATE_LIST_CACHE_LRU[old_key]

    _ACCUMULATE_LIST_CACHE_LRU[cache_key] = result

    return result


def get_selection_indices_for_tnd_softmax_update(t, n, sub_seq_len):
    device = torch.npu.current_device()
    cache_key = (t, n, tuple(map(int, sub_seq_len)))

    if cache_key in _SOFTMAX_INDICES_CACHE_LRU:
        return _SOFTMAX_INDICES_CACHE_LRU[cache_key]

    if _SOFTMAX_INDICES_CACHE_LRU:
        oldest_key = list(_SOFTMAX_INDICES_CACHE_LRU.keys())[0]
        if oldest_key != cache_key:
            # 删除旧缓存对象
            old_tensor = _SOFTMAX_INDICES_CACHE_LRU.pop(oldest_key)
            del old_tensor
            # 清理设备缓存
            if hasattr(torch_npu, 'empty_cache'):
                torch_npu.empty_cache()

    indices_list = []
    seq_start = 0

    base_head_indices = torch.arange(n, device=device, dtype=torch.long).unsqueeze(1)  # [n, 1]
    for seq_len in sub_seq_len:
        head_offsets = base_head_indices * (2 * seq_len)
        local_range = torch.arange(seq_len, 2 * seq_len, device=device, dtype=torch.long)
        selected = (head_offsets + local_range).flatten() + seq_start
        indices_list.append(selected)
        seq_start += 2 * seq_len * n

    result = torch.cat(indices_list) if indices_list else torch.empty(0, dtype=torch.long, device=device)
    _SOFTMAX_INDICES_CACHE_LRU[cache_key] = result
    return result
