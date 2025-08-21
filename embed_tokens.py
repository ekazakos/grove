import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from peft import get_peft_model
from model.GROVE import GROVEForCausalLM
from train import setup_tokenizer_and_special_tokens, initialize_custom_layers_in_model,\
    initialize_custom_layers_in_global_encoder, interpolate_positional_embeddings, setup_lora_config


def parse_args():
    parser = argparse.ArgumentParser(description="Embed tokens for GROVE model")
    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--grove_weights", default="/home/grove_checkpoints/grove_ft_iground_ckpt.bin", type=str)
    parser.add_argument("--token_embeddings_path", default="/home/token_embeddings_video.pt", type=str)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--precision", default='bf16', type=str)
    parser.add_argument("--image_size", default=512, type=int, help="Image size for grounding image encoder")
    parser.add_argument("--model_max_length", default=1536, type=int)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--lora_r", default=0, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    return parser.parse_args()


def initialize_model(args, tokenizer):
    """ Initialize the GROVE model. """
    model_args = {k: getattr(args, k) for k in
                  ["det_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx"]}

    model = GROVEForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, **model_args)
    print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    initialize_custom_layers_in_model(model)

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model

def prepare_model_for_inference(model, args):
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

    return model

def embed_tokens(model, vocab_size):
    # Get the embedding layer
    embedding_layer = model.get_model().embed_tokens

    # Create a tensor to store the embeddings
    embeddings = torch.zeros((vocab_size, embedding_layer.embedding_dim), dtype=torch.bfloat16).cuda()

    # Embed each token
    for input_id in tqdm(range(vocab_size)):
        tensor_input_id = torch.LongTensor([input_id]).cuda()
        embeddings[input_id, :] = model.get_model().embed_tokens(tensor_input_id.unsqueeze(0))

    # Save the embeddings as a PyTorch tensor
    torch.save(embeddings.cpu(), args.token_embeddings_path)

if __name__ == "__main__":
    args = parse_args()
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer)
    model = prepare_model_for_inference(model, args)
    interpolate_positional_embeddings(model)

    lora_r = args.lora_r
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    model.to(dtype=torch.bfloat16, device=args.local_rank)
    model.eval()

    state_dict = torch.load(args.grove_weights)
    if lora_r == 0:
        print(f"Loading weights into GROVE from {args.grove_weights}.")
        model.load_state_dict(state_dict, strict=False)
    else:
        updated_state_dict = {}
        for key in state_dict.keys():
            updated_key = f"base_model.model.{key}"
            updated_state_dict[updated_key] = state_dict[key]
        model.load_state_dict(updated_state_dict, strict=True)
        print(f"Successfully loaded weights into GROVE from {args.grove_weights}.")

    print(f"Embedding tokens with tokenizer length {len(tokenizer)}...")
    embed_tokens(model, len(tokenizer))