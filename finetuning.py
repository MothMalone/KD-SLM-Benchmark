import os
import argparse
import random
import torch

from configs import dataset as DATA_CONFIG
from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from configs import distillation_config as DISTIL_CONFIG

from train.train_utils import train
from configs.configs_utils import update_config
from data.data_utils import (get_dataloader, get_distillation_dataloader)
from train.tools import (setup, setup_environ_flags, clear_gpu_cache)
from models.models_utils import (get_model, get_distillation_models, get_optimizer)

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset.file", type=str, required=True, help="Path to the dataset loader")
    parser.add_argument("--lr", type=float, default=1e-6, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size_training", type=int, default=4, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=4, help="Validation batch size")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory path")
    parser.add_argument("--distillation_config_model_name", type=str, help="Model name for distillation")
    parser.add_argument("--distillation", action="store_true", help="Enable distillation")
    parser.add_argument("--distillation_config_enable_fsdp", action="store_true", help="Enable FSDP for distillation")
    parser.add_argument("--distillation_config_pure_bf16", action="store_true", help="Use pure BF16 for distillation")
    parser.add_argument("--distillation_config_distil_factor", type=float, default=1.5, help="Distillation factor")
    parser.add_argument("--save_step", type=int, default=100, help="Save step")
    parser.add_argument("--f", type=int, default=1, help="method")
    return parser.parse_args()

def main():
    args = parse_args()

    train_config, fsdp_config, distil_config, data_config = TRAIN_CONFIG(), FSDP_CONFIG(), DISTIL_CONFIG(), DATA_CONFIG()
    update_config((train_config, fsdp_config, data_config), **vars(args))
    update_config((distil_config), isSubmodule=True, **vars(args))
    #print(train_config)
    #print(fsdp_config)
    #print(data_config)

    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or distil_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    else: rank = 0

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load Model and Tokenizer
    if train_config.distillation:
        distil_config.model_name = args.distillation_config_model_name  
        student_tokenizer, teacher_tokenizer, model = get_distillation_models(train_config, distil_config, fsdp_config, rank, vars(args))
    else:
        tokenizer, model = get_model(train_config, fsdp_config, rank, vars(args))
    if rank == 0: print(model)

    # Load Data
    data_config.encoder_decoder = train_config.encoder_decoder
    if train_config.distillation:
        train_dataloader, teacher_train_dataloader, eval_dataloader, teacher_eval_dataloader = get_distillation_dataloader(data_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank)
    else:
        train_dataloader, eval_dataloader = get_dataloader(data_config, train_config, tokenizer, rank)

    # Get the optimizer and learning rate scheduler
    optimizer = get_optimizer(model, train_config, fsdp_config)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_config.lr, epochs=train_config.num_epochs, steps_per_epoch=len(train_dataloader),
                                                    pct_start=train_config.pct_start, div_factor=train_config.div_factor, final_div_factor=train_config.final_div_factor)

    f = train_config.f
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        distil_config,
        data_config,
        teacher_train_dataloader if train_config.distillation else None,
        teacher_eval_dataloader if train_config.distillation else None,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp or distil_config.enable_fsdp else None,
        rank,
        f,
    )
    if rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

if __name__ == "__main__":
    main()