# set -x
# export VLLM_ATTENTION_BACKEND=XFORMERS
# ENGINE=${1:-vllm}


# model_path=Qwen/Qwen2.5-VL-7B-Instruct
# n_gpus=$(nvidia-smi -L | wc -l)

# train_batch_size=32
# ppo_mini_batch_size=$((4 * n_gpus))
# ppo_micro_batch_size_per_gpu=1
# log_prob_micro_batch_size_per_gpu=1
# n_agent=5

# # # test
# # train_batch_size=4
# # ppo_mini_batch_size=4
# # ppo_micro_batch_size_per_gpu=1
# # log_prob_micro_batch_size_per_gpu=1
# # n_agent=2

# tensor_model_parallel_size=2
# val_before_train=False
# search_url="http://0.0.0.0:8002/search"
# rm_url="http://0.0.0.0:8003/eval"
# max_turns=4
# project_name="vrag"
# experiment_name="SFT_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups_qwen2_5_vl_7b"

# export RAY_memory_usage_threshold=0.995
# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=./data/rag/slidevqa_train_crop.parquet \
#     data.val_files=./data/rag/overall_test_crop.parquet \
#     data.train_batch_size=$train_batch_size \
#     data.max_prompt_length=8192 \
#     data.max_response_length=2048 \
#     data.image_key=images \
#     actor_rollout_ref.model.path=$model_path \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
#     actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.01 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
#     actor_rollout_ref.actor.state_masking=True \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
#     actor_rollout_ref.rollout.name=$ENGINE \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     actor_rollout_ref.rollout.n=1 \
#     actor_rollout_ref.rollout.n_agent=$n_agent \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     reward_model.reward_manager='rm' \
#     reward_model.rm_url=$rm_url \
#     algorithm.kl_ctrl.kl_coef=0.0 \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console'] \
#     trainer.project_name=$project_name \
#     trainer.experiment_name=$experiment_name \
#     trainer.n_gpus_per_node=$n_gpus \
#     trainer.nnodes=1 \
#     trainer.save_freq=25 \
#     trainer.test_freq=25 \
#     trainer.total_epochs=1 \
#     trainer.resume_mode=disable \
#     trainer.val_before_train=$val_before_train \
#     retriever.url=$search_url \
#     max_turns=$max_turns $@


#!/bin/bash
set -x

ENGINE=${1:-vllm}
model_path=Qwen/Qwen2.5-VL-3B-Instruct
n_gpus=2
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# 显存与 NCCL 配置
export CUDA_VISIBLE_DEVICES=4,6
export RAY_memory_usage_threshold=0.95
export VLLM_USE_GUARDED_ALLOCATION=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export VLLM_DISABLE_NVML=1
export NCCL_IGNORE_DISABLED_DEVICES=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1  # 可试验性地屏蔽 point-to-point comm
export VLLM_DISABLE_NVML=1

export RAY_TMPDIR=/data3/ray_tmp
export RAY_memory_usage_threshold=0.995

# 优化 batch 设置以降低显存占用
train_batch_size=6
ppo_mini_batch_size=6
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
n_agent=1

tensor_model_parallel_size=1
val_before_train=False
search_url="http://0.0.0.0:8002/search"
rm_url="http://0.0.0.0:8003/eval"
max_turns=4
project_name="vrag"
experiment_name="SFT_w_crop_${n_gpus}_gpus_${max_turns}_maxturns_${n_agent}_ngroups_qwen2_5_vl_7b"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/data3/xiongyuqi/VRAG3/data/rag/slidevqa_train_crop_3.parquet \
    data.val_files=/data3/xiongyuqi/VRAG3/data/rag/overall_test_crop_3.parquet \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.image_key=images \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.n_agent=$n_agent \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager='rm' \
    reward_model.rm_url=$rm_url \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.resume_mode=disable \
    trainer.val_before_train=$val_before_train \
    retriever.url=$search_url \
    max_turns=$max_turns $@