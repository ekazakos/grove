"""
train.py - GROVE Training
"""
import os
import logging
import sys
import time
import tqdm
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import json
import argparse
import deepspeed
import numpy as np
import transformers
from functools import partial
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from torchvision.ops import generalized_box_iou_loss
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter

from model.GROVE import GROVEForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.model.multimodal_encoder.modeling_clip import SpatioTemporalConvAdapter as GlobalSpatioTemporalConvAdapter
from model.SAM.modeling.image_encoder import SpatioTemporalConvAdapter as GroundingSpatioTemporalConvAdapter

from dataset.dataset import custom_collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN, 
                         AverageMeter, ProgressMeter, dict_to_cuda, Summary, intersectionAndUnionGPU)

from dataset.video_grounding_datasets.ActivityNetEntities import ActivityNetEntitiesDataset
from dataset.video_grounding_datasets.VidSTG import VidSTGDataset
from dataset.video_grounding_datasets.HowTo100M import HowTo100MDataset


def parse_args(args):
    parser = argparse.ArgumentParser(description="GROVE Model Training")

    # Model-specific settings
    parser.add_argument("--version", default="MBZUAI/GLaMM-GCG") # MBZUAI/GLaMM-GranD-Pretrained
    parser.add_argument("--vision_pretrained", default="./checkpoints/sam_vit_h_4b8939.pth", type=str)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])
    parser.add_argument("--tune_mm_mlp_adapter", action="store_true")
    parser.add_argument("--freeze_mm_mlp_adapter", action="store_true")
    parser.add_argument("--mm_use_im_start_end", action="store_true", default=True)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--image_size", default=512, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--with_region", action="store_true", default=True)
    parser.add_argument("--mm_vision_select_layer", default=-2, type=int)
    parser.add_argument("--pretrain_mm_mlp_adapter", default="", type=str)
    parser.add_argument("--precision", default='bf16', type=str)

    # Dataset settings
    parser.add_argument("--dataset", default="HowToGround", choices=["HowToGround", "ActivityNetEntities", "VidSTG"], type=str)
    parser.add_argument("--video_dir", default="/home/HowTo100M_small", type=str)
    parser.add_argument("--train_ann_dir", default="/home/train_annotations/", type=str)
    parser.add_argument("--val_ann_dir", default="/home/val_annotations/", type=str)
    parser.add_argument("--train_keys", default="/home/train_keys.pkl", type=str)
    parser.add_argument("--val_keys", default="/home/val_keys.pkl", type=str)
    parser.add_argument("--frame_timestamps", default="/home/ActivityNetEntities/timestamps_metadata.json")
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--num_frames", default=8, type=int)

    # Training settings
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--grove_weights", default=None, type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--auto_resume", action="store_true")
    parser.add_argument("--weight", default="", type=str)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--giou_loss_weight", default=1.0, type=float)
    parser.add_argument("--temp_objectness_loss_weight", default=1.0, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # Evaluation settings
    parser.add_argument("--val_dataset", default="RefCOCOgRegVal", type=str,
                        help="Choose from: CocoCapVal, RefCOCOgRegVal, VisGenomeRegVal, RefCOCOgSegmVal, PsgGCGVal, "
                             "RefCocoGCGVal, FlickrGCGVal")
    parser.add_argument("--bbox_validation", action="store_true")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")

    # Experiment settings
    parser.add_argument("--log_base_dir", default="/home/grove_checkpoints", type=str)
    parser.add_argument("--exp_name", default="iGround", type=str)

    return parser.parse_args(args)


def initialize_environment(args):
    """ Set up logging and model directories. """
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if dist.get_rank() == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        return SummaryWriter(args.log_dir)
    return None


def setup_tokenizer_and_special_tokens(args):
    """ Load tokenizer and add special tokens. """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version, model_max_length=args.model_max_length, padding_side="right", use_fast=False
    )
    print('\033[92m' + "---- Initialized tokenizer from: {} ----".format(args.version) + '\033[0m')
    tokenizer.pad_token = tokenizer.unk_token

    if not args.pretrained:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True
            )
        # modifications specific for regions
        reg_tokens = ['<bbox>', '<point>']
        # Adding special tokens for pixel grounding
        detection_tokens = ['[DET]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        special_tokens = reg_tokens + detection_tokens + phrase_tokens
        tokenizer.add_tokens(special_tokens, special_tokens=True)
    else:
        if args.use_mm_start_end:
            tokenizer.add_tokens(
                [DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN], special_tokens=True
            )
        # Adding special tokens for pixel grounding
        detection_tokens = ['[DET]']
        tokenizer.add_tokens(detection_tokens, special_tokens=True)

    args.bbox_token_idx = tokenizer("<bbox>", add_special_tokens=False).input_ids[1]
    args.det_token_idx = tokenizer("[DET]", add_special_tokens=False).input_ids[0]
    args.bop_token_idx = tokenizer("<p>", add_special_tokens=False).input_ids[1]
    args.eop_token_idx = tokenizer("</p>", add_special_tokens=False).input_ids[1]

    return tokenizer


def initialize_custom_layers_in_model(model):
    """
    Initialize custom layers in the GROVE model:
    - Spatio-temporal adapters
    - Bbox prediction head
    - Temporal objectness head
    - Projection layer for language input embeds
    """
    grounding_enc_adapters = model.get_model().grounding_encoder.image_encoder.adapters
    in_channels, out_channels, kernel_size = (grounding_enc_adapters[0].conv3d.in_channels,
                                              grounding_enc_adapters[0].conv3d.out_channels,
                                              grounding_enc_adapters[0].conv3d.kernel_size)
    model.get_model().grounding_encoder.image_encoder.adapters = nn.ModuleList([
        GroundingSpatioTemporalConvAdapter(in_channels, out_channels, kernel_size)
        for _ in range(len(grounding_enc_adapters))])

    bbox_prediction_head = model.get_model().grounding_encoder.mask_decoder.bbox_prediction_head
    hidden_dim = bbox_prediction_head[0].in_features
    model.get_model().grounding_encoder.mask_decoder.bbox_prediction_head = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim), 
        nn.ReLU(), 
        nn.Linear(hidden_dim, 4))
    
    # model.get_model().grounding_encoder.mask_decoder.temporal_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8)
    
    if model.config.use_temp_objectness:
        temporal_objectness_head = model.get_model().grounding_encoder.mask_decoder.temporal_objectness_head
        model.get_model().grounding_encoder.mask_decoder.temporal_objectness_head = nn.Linear(
            temporal_objectness_head.in_features, temporal_objectness_head.out_features
        )

    # llm_projector = model.get_model().llm_projector
    # model.get_model().llm_projector = nn.Linear(llm_projector.in_features, llm_projector.out_features)


def initialize_model(args, tokenizer):
    """ Initialize the GROVE model. """
    model_args = {k: getattr(args, k) for k in
                  ["train_mask_decoder", "out_dim", "ce_loss_weight", "giou_loss_weight", "temp_objectness_loss_weight",
                   "det_token_idx", "vision_pretrained", "vision_tower", "use_mm_start_end", "mm_vision_select_layer",
                   "pretrain_mm_mlp_adapter", "tune_mm_mlp_adapter", "freeze_mm_mlp_adapter", "mm_use_im_start_end",
                   "with_region", "bbox_token_idx", "eop_token_idx", "bop_token_idx", "num_frames"]}
    model_args["num_level_reg_features"] = 4
    model_args["use_temp_objectness"] = args.dataset == "HowToGround"

    model = GROVEForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, **model_args
    )
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')
    
    initialize_custom_layers_in_model(model)

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


def initialize_custom_layers_in_global_encoder(vision_tower):
    """ Initialize custom layers in the global image encoder: spatio-temporal adapters."""
    global_enc_adapters = vision_tower.vision_tower.vision_model.encoder.adapters
    in_channels, out_channels, kernel_size = (global_enc_adapters[0].conv3d.in_channels, 
                                              global_enc_adapters[0].conv3d.out_channels, 
                                              global_enc_adapters[0].conv3d.kernel_size)
    vision_tower.vision_tower.vision_model.encoder.adapters = nn.ModuleList([
        GlobalSpatioTemporalConvAdapter(in_channels, out_channels, kernel_size)
        for _ in range(len(global_enc_adapters))])


# TODO: Write function for freezing and unfreezing specific modules
def prepare_model_for_training(model, tokenizer, args):
    # Enable input gradients
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # Initialize vision tower
    print(
        '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
            args.vision_tower
        ) + '\033[0m'
    )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    
    initialize_custom_layers_in_global_encoder(vision_tower)
    
    vision_tower.to(dtype=torch.bfloat16, device=args.local_rank)

    # Initialize GROVE model and adjust requires_grad
    # Freeze model and unfreeze specific modules
    for param in model.get_model().parameters():
        param.requires_grad = False

     # Set requires_grad based on LoRA training
    lora_r = args.lora_r
    # if lora_r == 0:
    #     for p in model.get_model().layers.parameters():
    #         p.requires_grad = True
        # for p in model.get_model().mm_projector.parameters():
        #     p.requires_grad = True

    # Configure conversation library
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # Configure LoRA if applicable
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)

    if not args.pretrained:
        model.get_model().initialize_grove_model(model.get_model().config)
    else:
        # for param in model.get_model().grounding_encoder.parameters():
        #     param.requires_grad = False
        # Unfreeze adapters
        for param in model.get_model().grounding_encoder.image_encoder.adapters.parameters():
            param.requires_grad = True
        if model.get_model().config.train_mask_decoder:
            model.get_model().grounding_encoder.mask_decoder.train()
            for param in model.get_model().grounding_encoder.mask_decoder.parameters():
                param.requires_grad = True
        for param in model.get_model().grounding_encoder.mask_decoder.bbox_prediction_head.parameters():
            param.requires_grad = True
        if model.config.use_temp_objectness:
            for param in model.get_model().grounding_encoder.mask_decoder.temporal_objectness_head.parameters():
                param.requires_grad = True
        # for param in model.get_model().grounding_encoder.mask_decoder.temporal_layer.parameters():
        #     param.requires_grad = True

        # Projection layer (LP)
        # model.get_model().text_hidden_fcs.train()
        for param in model.get_model().text_hidden_fcs.parameters():
            param.requires_grad = True

    # Set requires_grad for vision tower and mm projector
    # for p in vision_tower.parameters():
    #     p.requires_grad = False
    # Unfreeze adapters
    for p in vision_tower.vision_tower.vision_model.encoder.adapters.parameters():
        p.requires_grad = True
    for p in model.get_model().mm_projector.parameters(): # VL projection layer
        p.requires_grad = True
    # for p in model.get_model().llm_projector.parameters(): # LL projection layer
    #     p.requires_grad = True
    for p in model.lm_head.parameters():
        p.requires_grad = True
    for p in model.get_model().embed_tokens.parameters():
        p.requires_grad = True

    # Set requires_grad based on LoRA training
    # lora_r = args.lora_r
    # # if lora_r == 0:
    # #     for p in model.get_model().layers.parameters():
    # #         p.requires_grad = True
    #     # for p in model.get_model().mm_projector.parameters():
    #     #     p.requires_grad = True

    # # Configure conversation library
    # conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # # Configure LoRA if applicable
    # if lora_r > 0:
    #     lora_config = setup_lora_config(model, args)
    #     model = get_peft_model(model, lora_config)

    # Resize token embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Make certain modules trainable
    # set_trainable_modules(model)


def setup_lora_config(model, args):
    """ Configure LoRA settings for the model. """

    def find_proj_layers(model, target_modules):
        """ Identify projection layers in the model for LoRA adaptation. """
        linear_cls = torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if (isinstance(module, linear_cls) and all(
                    x not in name for x in ["grounding_encoder", "vision_tower", "mm_projector", "text_hidden_fcs"]
            ) and any(x in name for x in target_modules)):
                lora_module_names.add(name)
        return sorted(list(lora_module_names))

    # Extracting LoRA target modules
    lora_target_modules = args.lora_target_modules.split(",")
    lora_module_names = find_proj_layers(model, lora_target_modules)

    # Configuring LoRA
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=lora_module_names, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM"
    )
    return lora_config


def set_trainable_modules(model):
    """ Make specified modules in the model trainable. """
    trainable_modules = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "region_encoder"]
    for name, param in model.named_parameters():
        if any(module in name for module in trainable_modules):
            print(f"Making trainable: {name}, Shape: {param.shape}")
            param.requires_grad = True

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print('\033[92m' + "---- Total parameters: ----{}".format(total_params) + '\033[0m')
        print('\033[92m' + "---- Trainable parameters: ----{}".format(trainable_params) + '\033[0m')

    count_parameters(model)


def initialize_datasets_and_loaders(args, tokenizer):
    # world_size = torch.cuda.device_count()
    world_size = dist.get_world_size()
    args.distributed = world_size > 1

    # Common dataset arguments
    common_ds_args = {"tokenizer": tokenizer,
                      "global_image_encoder": args.vision_tower,
                      "epoch_samples": args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
                      "precision": args.precision, "image_size": args.image_size,
                      "num_classes_per_sample": args.num_classes_per_sample}

    if args.dataset == "HowToGround":
        train_keys = pickle.load(open(args.train_keys, 'rb'))
        train_dataset = HowTo100MDataset(**common_ds_args, validation=False, ann_dir=args.train_ann_dir, 
                                         video_dir=args.video_dir, keys=train_keys)
    elif args.dataset == "ActivityNetEntities":
        train_keys = json.load(open(os.path.join(args.train_ann_dir, 'split_ids_anet_entities.json'), 'r'))
        train_keys = train_keys["training"]
        frame_timestamps = json.load(open(args.frame_timestamps, 'r'))
        ann_path = os.path.join(args.train_ann_dir, 'anet_entities_cleaned_class_thresh50_trainval.json')
        train_dataset = ActivityNetEntitiesDataset(**common_ds_args, validation=False, video_dir=args.video_dir,
                                                   ann_path=ann_path, keys=train_keys, frame_timestamps=frame_timestamps)
    elif args.dataset == "VidSTG":
        train_dataset = train_dataset = VidSTGDataset(**common_ds_args, validation=False, video_dir=args.video_dir,
                                                      ann_path=args.train_ann_dir)

    # world_size = torch.cuda.device_count()
    # Summing lengths of all datasets
    total_length = len(train_dataset)
    print(f"Training with {total_length} examples.")
    # Calculate steps per epoch
    effective_batch_size = args.batch_size * args.grad_accumulation_steps * world_size
    steps_per_epoch = total_length // effective_batch_size
    # modify steps per epoch
    args.steps_per_epoch = steps_per_epoch

    # Validation dataset
    if args.dataset == "HowToGround":
        val_keys = pickle.load(open(args.val_keys, 'rb'))
        val_dataset = HowTo100MDataset(**common_ds_args, validation=True, ann_dir=args.val_ann_dir, 
                                       video_dir=args.video_dir, keys=val_keys)
    elif args.dataset == "ActivityNetEntities":
        val_keys = json.load(open(os.path.join(args.val_ann_dir, 'split_ids_anet_entities.json'), 'r'))
        val_keys = val_keys["validation"]
        ann_path = os.path.join(args.val_ann_dir, 'anet_entities_cleaned_class_thresh50_trainval.json')
        val_dataset = ActivityNetEntitiesDataset(**common_ds_args, validation=True, video_dir=args.video_dir, 
                                                ann_path=ann_path, keys=val_keys, frame_timestamps=frame_timestamps)
    elif args.dataset == "VidSTG":
        val_dataset = VidSTGDataset(**common_ds_args, validation=True, video_dir=args.video_dir,
                                    ann_path=args.val_ann_dir)

    return train_dataset, val_dataset


def setup_data_loaders(args, train_dataset, val_dataset, tokenizer):
    sampler_args = {"shuffle": True, "drop_last": False}
    train_loader_args = {"batch_size": args.batch_size, "shuffle": False, "num_workers": args.workers,
                         "pin_memory": False,} #"multiprocessing_context": "forkserver"}
    val_loader_args = {"batch_size": args.val_batch_size, "shuffle": False, "num_workers": args.workers,
                       "pin_memory": False, }#"multiprocessing_context": "forkserver"}
    collate_fn_args_train = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=False
    )
    inference_mode = args.bbox_validation
    collate_fn_args_val = partial(
        custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank,
        inference=inference_mode
    )

    # Training loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, sampler=torch.utils.data.distributed.DistributedSampler(
            train_dataset, **sampler_args
            ), collate_fn=collate_fn_args_train, **train_loader_args
        )

    # Validation loader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **val_loader_args, collate_fn=collate_fn_args_val,
        sampler=torch.utils.data.distributed.DistributedSampler(val_dataset, **sampler_args), )

    return train_loader, val_loader


def initialize_deepspeed(model, tokenizer, args):
    ds_config = {"train_micro_batch_size_per_gpu": args.batch_size,
                 "gradient_accumulation_steps": args.grad_accumulation_steps,
                 "optimizer": {"type": "AdamW", "params": {"lr": args.lr, "weight_decay": args.wd,
                                                           "betas": (args.beta1, args.beta2)}},
                 "scheduler": {"type": "WarmupDecayLR",
                               "params": {"total_num_steps": args.epochs * args.steps_per_epoch, "warmup_min_lr": 0,
                                          "warmup_max_lr": args.lr, "warmup_num_steps": 100, "warmup_type": "linear"}},
                 "fp16": {"enabled": args.precision == "fp16"}, "bf16": {"enabled": args.precision == "bf16"},
                 "gradient_clipping": 1.0,
                 "zero_optimization": {"stage": 2, "contiguous_gradients": True, "overlap_comm": True,
                                       "reduce_scatter": True, "reduce_bucket_size": 5e8,
                                       "allgather_bucket_size": 5e8}, }

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), collate_fn=partial(
            custom_collate_fn, tokenizer=tokenizer, use_mm_start_end=args.use_mm_start_end, local_rank=args.local_rank
        ), config=ds_config
    )

    return model_engine, optimizer, scheduler


def resume_training_from_checkpoint(model_engine, args):
    if args.auto_resume and not args.resume:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")


def resize_abs_pos_embedding(pos_embed, target_size, patch_size, embed_dim):
    """
    Resize position embedding directly to a new target size using bicubic interpolation.
    
    Parameters:
    - pos_embed: The original position embedding tensor in the shape of 
      (1, img_size // patch_size, img_size // patch_size, embed_dim).
    - target_size: The new target size for the image (height/width, assuming square images).
    - patch_size: The size of patches the image was divided into.
    - embed_dim: The dimension of the embedding.
    
    Returns:
    - Resized position embedding tensor.
    """
    # Calculate the new height and width based on the target size and patch size
    new_h, new_w = target_size // patch_size, target_size // patch_size
    
    # Reshape position embedding to (batch_size, channels, height, width) format for interpolation
    pos_embed_reshaped = pos_embed.permute(0, 3, 1, 2)  # Move the embedding dimension to the channel dimension
    
    # Perform bicubic interpolation
    pos_embed_resized = F.interpolate(pos_embed_reshaped, size=(new_h, new_w), mode='bicubic', align_corners=False)
    
    # Reshape back to the original position embedding format
    pos_embed_resized = pos_embed_resized.permute(0, 2, 3, 1)  # Move the channels back to the embedding dimension
    
    return pos_embed_resized


def resize_rel_pos_embedding(pos_embed_h, pos_embed_w, target_size, patch_size, embed_dim):
    """
    Resize position embedding directly to a new target size using bicubic interpolation.
    
    Parameters:
    - pos_embed: The original position embedding tensor in the shape of 
      (1, img_size // patch_size, img_size // patch_size, embed_dim).
    - target_size: The new target size for the image (height/width, assuming square images).
    - patch_size: The size of patches the image was divided into.
    - embed_dim: The dimension of the embedding.
    
    Returns:
    - Resized position embedding tensor.
    """
    # Calculate the new height and width based on the target size and patch size
    new_h, new_w = 2 * (target_size // patch_size) - 1, 2 * (target_size // patch_size) - 1
    
    # Reshape position embedding to (batch_size, channels, height, width) format for interpolation
    pos_embed_h_reshaped = pos_embed_h.unsqueeze(0).unsqueeze(0).permute(0, 3, 2, 1)  # Move the embedding dimension to the channel dimension
    pos_embed_w_reshaped = pos_embed_w.unsqueeze(0).unsqueeze(0).permute(0, 3, 1, 2)  # Move the embedding dimension to the channel dimension
    # Perform bicubic interpolation
    pos_embed_h_resized = F.interpolate(pos_embed_h_reshaped, size=(new_h, 1), mode='bicubic', align_corners=True)
    pos_embed_w_resized = F.interpolate(pos_embed_w_reshaped, size=(1, new_w), mode='bicubic', align_corners=True)
    # Reshape back to the original position embedding format
    pos_embed_h_resized = pos_embed_h_resized.permute(0, 3, 2, 1).squeeze(0).squeeze(0)  # Move the channels back to the embedding dimension
    pos_embed_w_resized = pos_embed_w_resized.permute(0, 2, 3, 1).squeeze(0).squeeze(0)  # Move the channels back to the embedding dimension
    return pos_embed_h_resized, pos_embed_w_resized 


def interpolate_positional_embeddings(ds_model):
    # Resize absolute position embeddings
    abs_pos_embed = ds_model.model.grounding_encoder.image_encoder.pos_embed.clone().contiguous()
    abs_pos_embed_new = resize_abs_pos_embedding(abs_pos_embed, 512, 16, abs_pos_embed.shape[3])
    ds_model.model.grounding_encoder.image_encoder.pos_embed = nn.Parameter(abs_pos_embed_new.contiguous())

    # Resize relative position embeddings
    encoder_global_attn_indexes=[7, 15, 23, 31]
    for i in encoder_global_attn_indexes:
        rel_pos_h = ds_model.model.grounding_encoder.image_encoder.blocks[i].attn.rel_pos_h.clone().contiguous()
        rel_pos_w = ds_model.model.grounding_encoder.image_encoder.blocks[i].attn.rel_pos_w.clone().contiguous()
        rel_pos_h_new, rel_pos_w_new = resize_rel_pos_embedding(rel_pos_h, rel_pos_w, 512, 16, rel_pos_h.shape[1])
        ds_model.model.grounding_encoder.image_encoder.blocks[i].attn.rel_pos_h = nn.Parameter(rel_pos_h_new.contiguous())
        ds_model.model.grounding_encoder.image_encoder.blocks[i].attn.rel_pos_w = nn.Parameter(rel_pos_w_new.contiguous())

    ds_model.model.grounding_encoder.image_encoder.img_size = 512


def setup_logger():
    
    # Get the rank of the current process
    rank = dist.get_rank()

    # Create a logger for this rank
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.DEBUG)  # Set the desired logging level

    # Create a file handler that logs to a file specific to this rank
    log_filename = f"train_log_rank_{rank}.log"
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)

    # Create a console handler to output to stdout (optional)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(f'%(asctime)s - Rank {rank} - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def main(args):

    logger = setup_logger()

    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    prepare_model_for_training(model, tokenizer, args)
    # Perform interpolation on the model's positional embeddings within the DeepSpeed engine
    interpolate_positional_embeddings(model)
    model.to(dtype=torch.bfloat16, device=args.local_rank)

    # Fine-tune
    if args.grove_weights:
        print(f"Fine-tuning using GROVE weights from {args.grove_weights}.")
        state_dict = torch.load(args.grove_weights)
        model.load_state_dict(state_dict, strict=False)

    train_dataset, val_dataset = initialize_datasets_and_loaders(args, tokenizer)
    model_engine, optimizer, scheduler = initialize_deepspeed(model, tokenizer, args)

    # # Perform interpolation on the model's positional embeddings within the DeepSpeed engine
    # ds_model = model_engine.module
    # interpolate_positional_embeddings(ds_model)

    resume_training_from_checkpoint(model_engine, args)

    train_loader, val_loader = setup_data_loaders(args, train_dataset, val_dataset, tokenizer)
    dataset_iter = iter(train_loader)

    writer = initialize_environment(args)

    if args.eval_only:
        cur_val_loss = validate_model_performance(val_loader, model_engine, 0, writer, args)[0]
        exit()

    # epoch_seeds = [random.randint(0, 100000) for _ in range(args.epochs)]

    if args.dataset == "HowToGround":
        best_giou, best_temp_objectness_acc, best_val_loss = 0.0, 0.0, np.inf
    else:
        best_giou, best_val_loss = 0.0, np.inf
    # try:
    for epoch in range(args.start_epoch, args.epochs):
        # random.seed(epoch_seeds[epoch])

        dataset_iter = train(train_loader, model_engine, epoch, scheduler, writer, dataset_iter, args, logger)
        dist.barrier()

        if args.bbox_validation:
            giou, temp_objectness_acc = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            dist.barrier()

            is_best = giou > best_giou
            best_giou = max(giou, best_giou)
            if temp_objectness_acc is not None:
                best_temp_objectness_acc = temp_objectness_acc if is_best else best_temp_objectness_acc
                if dist.get_rank() == 0:  # Log the progress
                    print((f"Epoch: {epoch}, giou: {giou}, temp objectness acc: {temp_objectness_acc}, "
                        "best_giou: {best_giou}, best_temp_objectness_acc: {best_temp_objectness_acc}"))
                save_checkpoint(model_engine, tokenizer, args, epoch, 'giou-temp_objectness_acc', f"{giou:.4f}-{temp_objectness_acc:.4f}", is_best)
            else:
                if dist.get_rank() == 0:
                    print((f"Epoch: {epoch}, giou: {giou}, best_giou: {best_giou}"))
                save_checkpoint(model_engine, tokenizer, args, epoch, 'giou', f"{giou:.4f}", is_best)
        else:
            cur_val_loss = validate_model_performance(val_loader, model_engine, epoch, writer, args)
            dist.barrier()
            is_best = cur_val_loss < best_val_loss
            best_val_loss = min(cur_val_loss, best_val_loss)
            if dist.get_rank() == 0:  # Log the progress
                print(f"Epoch: {epoch}, Current Validation Loss: {cur_val_loss:.4f}, Best Validation Loss: {best_val_loss:}")
            save_checkpoint(model_engine, tokenizer, args, epoch, 'loss', f"{cur_val_loss:.4f}", is_best)
    # except Exception as e:
        # logger.exception(f"An exception occurred on rank {dist.get_rank()}: {e}")


def save_checkpoint(model_engine, tokenizer, args, epoch, metric_name, metric_value, is_best):
    """ Saves the model checkpoint. """
    # If the checkpoint is the best, save it in ckpt_model_best, else in ckpt_model_last_epoch
    # save_dir_name = "ckpt_model_best" if is_best else "ckpt_model_last_epoch"
    if is_best:
        save_dir_name = "ckpt_model_best"
        save_dir = os.path.join(args.log_dir, save_dir_name)
        # Ensure the directory exists
        if dist.get_rank() == 0:
            os.makedirs(save_dir, exist_ok=True)
            ckpt_filename = f"epoch_{epoch}_val_{metric_name}_{metric_value}.pth"
            torch.save({"epoch": epoch, f"val_{metric_name}": metric_value}, os.path.join(save_dir, ckpt_filename))
        dist.barrier()
        # model_to_save = model_engine.module if hasattr(model_engine, "module") else model_engine
        # model_to_save.save_pretrained(save_dir)
        # tokenizer.save_pretrained(save_dir)
        model_engine.save_checkpoint(save_dir)


def train(data_loader, model, epoch, scheduler, writer, dataset_iter, args, logger):
    """Main training loop."""

    def get_next_input(iterator, data_loader):
        """Retrieve next input from the iterator, or reinitialize if necessary."""
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(data_loader)
            return next(new_iterator), new_iterator

    def log_progress():
        """Log training progress."""
        if global_step % args.print_freq == 0:
            dist.barrier()
            if args.distributed:
                for tracker in trackers.values():
                    tracker.all_reduce()

            if dist.get_rank() == 0:
                progress.display(global_step + 1)
                for key, tracker in trackers.items():
                    writer.add_scalar(f"train/{key}", tracker.avg, global_step)
                writer.add_scalar("metrics/total_secs_per_batch", batch_time.avg, global_step)
                writer.add_scalar("metrics/data_secs_per_batch", data_time.avg, global_step)

            for tracker in trackers.values():
                tracker.reset()

    batch_time = AverageMeter("Time", ":.4f")
    data_time = AverageMeter("Data", ":.4f")
    trackers = {"loss": AverageMeter("Loss", ":.4f"),
                "ce_loss": AverageMeter("CELoss", ":.4f"),
                "giou_loss": AverageMeter("GIoULoss", ":.4f"),
                "l1_loss": AverageMeter("L1Loss", ":.4f"),}
    progress = ProgressMeter(args.steps_per_epoch, list(trackers.values()), prefix=f"Epoch: [{epoch}]")

    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            # Select data loader based on step choice
            data_batch, new_iter = get_next_input(dataset_iter, data_loader)
            dataset_iter = new_iter

            data_time.update(time.time() - end)
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()

            # logger.info(f"Shape of global_enc_images: {data_batch['global_enc_images'].shape}")
            # logger.info(f"Shape of grounding_enc_images: {data_batch['grounding_enc_images'].shape}")
            # logger.info(f"Shape of input_ids: {data_batch['input_ids'].shape}")
            # logger.info(f"Conversation list: {data_batch['conversation_list']}")
            # logger.info(f"Temp objectness: {data_batch['temp_objectness_labels_list']}")

            output_dict = model(**data_batch)

            # Update training metrics
            for key, tracker in trackers.items():
                if key in output_dict:
                    # tracker.update(output_dict[key].item(), data_batch["grounding_enc_images"].size(0))
                    tracker.update(output_dict[key].item(), 1)

            # try:
            model.backward(output_dict["loss"])
            # except Exception as e:
            #     logger.exception(f"Exception during backward pass on rank {dist.get_rank()}: {e}")
            #     # for name, param in model.named_parameters():
            #     #     if param.grad is not None:
            #     #         # print(f"Rank {rank}, Parameter: {name}, Gradient shape: {param.grad.shape}", flush=True)
            #     #         logger.info(f"Rank {dist.get_rank()}, Parameter: {name}, Gradient shape: {param.grad.shape}, Param numel: {param.numel()}")
            #     #     else:
            #     #         # print(f"Rank {rank}, Parameter: {name}, Gradient is None", flush=True)
            #     #         logger.info(f"Rank {dist.get_rank()}, Parameter: {name}, Gradient is None, Param numel: {param.numel()}")
            #     logger.info(f"Batch key: {'temp_objectness_labels_list'}, Batch shape: {[len(l) for l in data_batch['temp_objectness_labels_list']]}")
            #     raise e
            model.step()

        batch_time.update(time.time() - end)
        end = time.time()
        log_progress()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if dist.get_rank() == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return dataset_iter


def validate_model_performance(validation_loader, training_model, current_epoch, tensorboard_writer, args):
    if args.bbox_validation:
        if args.dataset == "HowToGround":
            trackers = {"gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM),
                    "temp_objectness_acc": AverageMeter("temp_objectness_acc", ":.4f", Summary.SUM),}
        else:
            trackers = {"gIoU": AverageMeter("gIoU", ":.4f", Summary.SUM),}

        training_model.eval()
        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                results = training_model(**data_batch)

            pred_bboxes = results["pred_bboxes"]
            gt_bboxes = results["gt_bboxes"].int()
            logits_temp_objectness = results["logits_temp_objectness"]
            gt_temp_objectness = results["gt_temp_objectness"].int()
            assert len(predictions) == 1
            if logits_temp_objectness is not None:
                giou_sum, temp_objectness_sum = 0.0, 0.0
                num_bboxes = 0
                num_max_bboxes = 0

                for batch_idx, (pred_bboxes_video, logits_temp_objectness_video)  in enumerate(zip(pred_bboxes, logits_temp_objectness)):
                    for frame_idx, (pred_bboxes_frame, logits_temp_objectness_frame) in enumerate(zip(pred_bboxes_video, logits_temp_objectness_video)):
                        gt_bboxes_frame = gt_bboxes[batch_idx][frame_idx]
                        gt_temp_objectness_frame = gt_temp_objectness[batch_idx][frame_idx]
                        
                        giou_sum += generalized_box_iou_loss(pred_bboxes_frame[gt_temp_objectness_frame.bool()], 
                                                            gt_bboxes_frame, 
                                                            reduction="sum")
                        
                        pred_temp_objectness_frame = (F.sigmoid(logits_temp_objectness_frame) > 0.5).int()
                        temp_objectness_sum += (pred_temp_objectness_frame == gt_temp_objectness_frame).sum().item()

                        num_bboxes += gt_bboxes_frame.shape[0]
                        num_max_bboxes += pred_bboxes_frame.shape[0]
            else:
                giou_sum = 0.0
                num_bboxes = 0
            
                for batch_idx, (pred_bboxes_video)  in enumerate(pred_bboxes):
                    for frame_idx, (pred_bboxes_frame) in enumerate(pred_bboxes_video):
                        gt_bboxes_frame = gt_bboxes[batch_idx][frame_idx]
                        gt_temp_objectness_frame = gt_temp_objectness[batch_idx][frame_idx]
                        
                        giou_sum += generalized_box_iou_loss(pred_bboxes_frame[gt_temp_objectness_frame.bool()], 
                                                            gt_bboxes_frame, 
                                                            reduction="sum")
                        
                        num_bboxes += gt_bboxes_frame.shape[0]

            trackers["gIoU"].update(giou_sum, n=num_bboxes)
            if logits_temp_objectness is not None:
                trackers["temp_objectness_acc"].update(temp_objectness_sum, n=num_max_bboxes)

        for meter in trackers.values():
            meter.all_reduce()

        giou = trackers["gIoU"].avg
        if args.dataset == "HowToGround":
            temp_objectness_acc = trackers["temp_objectness_acc"].avg

        if dist.get_rank() == 0:
            tensorboard_writer.add_scalar("val/giou", giou, current_epoch)
            print("giou: {:.4f}".format(giou))
            if args.dataset == "HowToGround":
                tensorboard_writer.add_scalar("val/temp_objectness_acc", temp_objectness_acc, current_epoch)
                print("temp objectness acc: {:.4f}".format(temp_objectness_acc))

        if args.dataset == "HowToGround":
            return giou, temp_objectness_acc
        else:
            return giou
    else:
        # Initializing performance trackers
        trackers = {"loss": AverageMeter("Loss", ":.4f"), "ce_loss": AverageMeter("CELoss", ":.4f"),
                    "giou_loss": AverageMeter("GIoULoss", ":.4f"),
                    "l1_loss": AverageMeter("L1Loss", ":.4f"),}
        if args.dataset == "HowToGround":
            trackers["temp_objectness_loss"] = AverageMeter("TempObjectnessLoss", ":.4f")

        # Prepare model for validation phase
        # Hack to get the loss
        training_model.train()

        for data_batch in tqdm.tqdm(validation_loader):
            # Prepare data and convert relevant tensors to bfloat16
            data_batch = dict_to_cuda(data_batch)
            for key in ["global_enc_images", "grounding_enc_images"]:
                data_batch[key] = data_batch[key].bfloat16()
            torch.cuda.empty_cache()
            # Model inference without gradient tracking
            with torch.no_grad():
                predictions = training_model(**data_batch)
            # Update performance metrics)
            for key, tracker in trackers.items():
                # tracker.update(predictions[key].item(), data_batch["grounding_enc_images"].size(0))
                tracker.update(predictions[key].item(), 1)

        # Synchronize metrics across processes
        for tracker in trackers.values():
            tracker.all_reduce()
        # Calculate average validation loss
        if args.dataset == "HowToGround":
            avg_val_loss = (1/3)*(trackers["giou_loss"].avg + trackers["l1_loss"].avg + trackers["temp_objectness_loss"].avg)
        else:
            avg_val_loss = (1/2)*(trackers["giou_loss"].avg + trackers["l1_loss"].avg)
        # Tensorboard logging for primary process
        if dist.get_rank() == 0:
            for key, tracker in trackers.items():
                tensorboard_writer.add_scalar(f"val/{key}", tracker.avg, current_epoch)
            # tensorboard_writer.add_scalar("val/loss", avg_val_loss, current_epoch)

        return avg_val_loss

def set_seed(seed):
    # Set the seed for torch
    torch.manual_seed(seed)
    # If using GPUs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Set the seed for numpy
    np.random.seed(seed)
    # Set the seed for python random
    random.seed(seed)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    deepspeed.init_distributed()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    set_seed(42)

    main(args)
