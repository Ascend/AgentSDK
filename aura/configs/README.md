# 配置文件以及配置文件命名规范说明

## infer配置文件命名规范

    推理侧只感知到是否PD分离，不感知训练的配置
    推理引擎名称_infer_p卡数d卡数（推理PD卡数，未开pd分离则填: i卡数）_模型名称_{扩展字段|可选}.yaml
    举例如下：
    PD分离：
    vllm_infer_p8d8_qwen25_7b.yaml
    PD混合：
    sglang_infer_i8_qwen25_7b.yaml

## train配置文件命名规范

    训练侧只感知训推分离，不感知推理侧的配置，也不感知推理是否PD分离
    train/msrl_conf 目录下的配置文件为摸版文件，每个模型一个配置文件，这里面的配置文件严禁修改
    需要修改的文件在train目录下，命名格式如下：
    训练引擎名称_train_异步分离或共卡（async|hybrid）_t卡数（训练卡数）_模型名称_工具名称_{扩展字段|可选}.yaml
    举例如下：
    训推分离：
    msrl_train_async_t8_qwen25_7b_dtn_code.yaml
    verl_train_async_t8_qwen25_7b_dtn_code.yaml
    训推共卡：
    msrl_train_hybrid_t8_qwen25_7b_dtn_code.yaml
    测试使用：比如只启动rollout推理, train mock掉
    msrl_train_async_t8_qwen25_7b_dtn_code_dummy_train.yaml

## base配置文件

    只配置基础的工作模式，infer配置文件名，train配置文件名等

## hosts配置文件

    炼丹炉和云道不需要配置该文件
    该文件提供给线下手动调试时, 机器手动分配VC_TASK_INDEX
