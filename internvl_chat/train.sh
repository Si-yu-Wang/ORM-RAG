# Using 8 GPUs, fine-tune the full LLM, cost about 30G per GPU
source ~/.bashrc
# GPUS=4 PER_DEVICE_BATCH_SIZE=4 sh shell/report/new/baseline_align.sh
GPUS=2 PER_DEVICE_BATCH_SIZE=2 sh shell/report/new/baseline_frame_10.sh

# # Using 2 GPUs, fine-tune the LoRA, cost about 27G per GPU
# GPUS=2 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh
# # Using 8 GPUs, fine-tune the LoRA, cost about 27G per GPU
# GPUS=8 PER_DEVICE_BATCH_SIZE=1 sh shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora.sh