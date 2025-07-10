#export HOME = ""

export CUDA_VISIBLE_DEVICES=0 python finetuning.py \
--model_name $HOME/Multi-Level-OT/EleutherAI/pythia-410m-deduped \
--dataset.file $HOME/Multi-Level-OT/llm_distillation/datasets/loader/squad.py \
--lr 1e-6 \
--num_epochs 5 \
--batch_size_training 2 \
--val_batch_size 2 \
--output_dir $HOME/Multi-Level-OT/output2 \
--distillation_config_model_name $HOME/models/meta-llama/Llama-2-7b-chat-hf \
--distillation \
--distillation_config_enable_fsdp \
--distillation_config_pure_bf16 \
--distillation_config_distil_factor 1.5 \
--save_step 2000
--f 5
