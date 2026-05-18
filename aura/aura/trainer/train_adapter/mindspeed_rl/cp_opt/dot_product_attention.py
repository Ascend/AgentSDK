from megatron.core import mpu
from mindspeed.core.parallel_state import (
    get_context_parallel_group_for_hybrid_ring,
    get_context_parallel_for_hybrid_ring_world_size,
    get_context_parallel_for_hybrid_ring_rank,
    get_context_parallel_for_hybrid_ring_global_ranks,
)
from mindspeed_llm.core.transformer.dot_product_attention import (
    get_args,
    ringattn_context_parallel,
    adaptive_attn_context_parallel,
)
from mindspeed.core.parallel_state import (
    get_ring_ranks_for_intra_window,
    get_ring_ranks_for_inter_window_kv,
    get_ring_ranks_for_inter_window_dkv,
    get_ring_group_for_intra_window,
    get_ring_group_for_intra_window_send_recv_overlap,
)
from mindspeed.core.context_parallel.utils import get_scheduling_info


def do_ring_context_parallel(
    q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0.0, pse=None, pse_type=None, packed_seq_params=None
):
    args = get_args()
    if args.shape_order == "TND":
        # Convert cu_seqlens_q and cu_seqlens_kv to tensors if needed (for .tolist() compatibility)
        if packed_seq_params is not None:
            if hasattr(packed_seq_params, 'cu_seqlens_q'):
                if not isinstance(packed_seq_params.cu_seqlens_q, torch.Tensor):
                    packed_seq_params.cu_seqlens_q = torch.tensor(
                        packed_seq_params.cu_seqlens_q, dtype=torch.int64, device=torch.cuda.current_device()
                    )
            if hasattr(packed_seq_params, 'cu_seqlens_kv'):
                if not isinstance(packed_seq_params.cu_seqlens_kv, torch.Tensor):
                    packed_seq_params.cu_seqlens_kv = torch.tensor(
                        packed_seq_params.cu_seqlens_kv, dtype=torch.int64, device=torch.cuda.current_device()
                    )

    in_hybrid_mode = get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None
    if in_hybrid_mode:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()
    else:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()

    cp_para = dict()

    cp_para['causal'] = args.cp_attention_mask_type == 'causal'
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    if args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
        cp_para['cp_global_ranks'] = cp_global_ranks
        cp_para['cp_group_for_send_recv_overlap'] = (
            mpu.get_context_parallel_group_for_send_recv_overlap() if args.use_cp_send_recv_overlap else None
        )
        cp_para['pse'] = pse
        cp_para['pse_type'] = pse_type

        cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
        cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
        cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
        cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
        cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()

        output = ringattn_context_parallel(
            q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p, packed_seq_params
        )
    else:
        cp_para['scheduling_info'] = get_scheduling_info()
        output = adaptive_attn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p)
    return output
