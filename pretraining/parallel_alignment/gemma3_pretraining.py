# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Gemma3 language model pretrain"""

import torch
from lightning.pytorch.loggers import WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.model.gemma3 import Gemma3Config12B, Gemma3Model
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler, MegatronOptimizerModule
from nemo.utils.exp_manager import TimingCallback
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning.pytorch.strategies.utils import RestoreConfig

from configure_data import get_data_parameters

from argparse import ArgumentParser
from datetime import timedelta


def main(args):
    """Entrypoint"""
    tokenizer = AutoTokenizer(
        pretrained_model_name="/tokenizer"
    )

    data = llm.PreTrainingDataModule(
        tokenizer = tokenizer,
        seq_length = args.seq_length,
        global_batch_size = args.global_batch_size,
        micro_batch_size = args.micro_batch_size,
        reset_position_ids = True,
        reset_attention_mask = True,
        eod_mask_loss=True,
        **get_data_parameters()
    )

    config_kwargs = {}
    if args.enable_activation_checkpointing:
        config_kwargs["recompute_granularity"] = "full"
        config_kwargs["recompute_method"] = "block"
        config_kwargs["recompute_num_layers"] = 48

    model_config = Gemma3Config12B(**config_kwargs)
    model = Gemma3Model(model_config)

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size,
        pipeline_dtype=torch.bfloat16,
        virtual_pipeline_model_parallel_size=None,
        context_parallel_size=args.cp_size,
        expert_model_parallel_size=1,
        sequence_parallel=args.use_sequence_parallelism,
        gradient_as_bucket_view=True,
        ckpt_async_save=True,
        ckpt_parallel_save=True,
        ckpt_parallel_load=True,
        ckpt_parallel_save_optim=True,
        ckpt_load_strictness="log_all",
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
        ),
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus_per_node,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        limit_val_batches=args.limit_val_batches,
        limit_test_batches=args.limit_test_batches,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        strategy=strategy,
        accumulate_grad_batches=1,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(precision=args.precision),
        enable_checkpointing=False,
        callbacks=[
            TimingCallback(),
        ],
    )

    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.max_lr,
        weight_decay=0.1,
        bf16=True,
        fp16=False,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=args.gradient_clip_val,
    )
    lr_scheduler = CosineAnnealingScheduler(
        warmup_steps=args.warmup_steps,
        constant_steps=args.constant_steps,
        min_lr=args.min_lr,
    )
    opt = MegatronOptimizerModule(config=opt_config, lr_scheduler=lr_scheduler)

    ckpt = nl.ModelCheckpoint(
        save_top_k=3,
        save_last=True,
        save_optim_on_train_end=False,
        filename="{val_loss:.2f}-{step}-{consumed_samples}",
        train_time_interval=timedelta(minutes=30)
    )
    wandb = WandbLogger(
        name=args.wandb_name,
        save_dir=args.experiment_dir,
        project=args.wandb_project,
        offline=True
    )
    logger = nl.NeMoLogger(
        log_dir=args.experiment_dir,
        version=args.version,
        log_global_rank_0_only=True,
        update_logger_directory=False,
        ckpt=ckpt,
        wandb=wandb
    )

    restore_config = RestoreConfig(
        path = args.model_path,
        load_model_state=True,
        load_optim_state=False,
        load_artifacts=True
    )
    resume = nl.AutoResume(
        restore_config = restore_config,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True
    )

    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        resume=resume,
        optim=opt,
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--enable_activation_checkpointing", action="store_true")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--wandb_name", type=str, required=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--tp_size", type=int, default=8)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--use_sequence_parallelism", action="store_true")
    parser.add_argument("--num_nodes", type=int, default=2)
    parser.add_argument("--num_gpus_per_node", type=int, default=4)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--limit_test_batches', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=int, default=1)
    parser.add_argument('--log_every_n_steps', type=int, default=10)
    parser.add_argument('--val_check_interval', type=int, default=100)
    parser.add_argument('--global_batch_size', type=int, default=512)
    parser.add_argument('--micro_batch_size', type=int, default=1)
    parser.add_argument('--seq_length', type=int, default=8192)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--constant_steps', type=int, default=0)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--max_lr', type=float, default=5e-5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
