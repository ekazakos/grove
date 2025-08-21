import os
import re
import cv2
import json
import pickle
import bleach
import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
import transformers
from transformers import AutoTokenizer, CLIPImageProcessor
from peft import get_peft_model
import ffmpeg

from model.GROVE import GROVEForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.SAM.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VIDEO_TOKEN, IMAGE_TOKEN_INDEX
import random
from train import setup_tokenizer_and_special_tokens, initialize_custom_layers_in_model,\
    initialize_custom_layers_in_global_encoder, interpolate_positional_embeddings, setup_lora_config


def parse_args():
    parser = argparse.ArgumentParser(description="GROVE Inference - iGround")

    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--grove_weights", default="/home/grove_checkpoints/grove_ft_iground_ckpt.bin", type=str)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--video_dir", default="/home/HowTo100M/", type=str, help="Path to the HowTo100M video directory")
    parser.add_argument("--video_info", default="/home/iGround_annotations/iGround_test_set_processed.pkl")
    parser.add_argument("--output_path", type=Path, default=Path("/home/grove_inference_output/result.pkl"))
    parser.add_argument("--token_embeddings", default="/home/token_embeddings_video.pt", type=str)
    parser.add_argument("--image_size", default=512, type=int, help="image size")
    parser.add_argument("--num_frames", default=8, type=int)
    parser.add_argument("--temp_objectness_threshold", default=0.5, type=float)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--conv_type", default="llava_v1", type=str, choices=["llava_v1", "llava_llama_2"])

    # DDP Related parameters
    parser.add_argument("--batch_size", required=False, default=1)
    parser.add_argument("--num_workers", default=7, type=int)
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)

    parser.add_argument("--lora_r", default=0, type=int, help="LoRA rank")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)

    return parser.parse_args()


def tokenize_prompt(tokenizer, use_mm_start_end, answer=""):
    # Prepare prompt
    conv = conversation_lib.conv_templates[args.conv_type].copy()
    conv.messages = []

    instructions = "Could you please give me a description of the video? Please respond with interleaved bounding boxes for the corresponding parts of the answer."
    instructions = bleach.clean(instructions)
    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')

    instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')
    prompt = instructions
    prompt = f"The {DEFAULT_VIDEO_TOKEN} provides an overview of the video." + "\n" + prompt
    if use_mm_start_end:
        replace_token = (DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN)
        prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], answer)
    prompt = conv.get_prompt()

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")

    return input_ids

def update_and_sort_video_outputs(gathered_results):
    """
    Updates the video_outputs dict with gathered results from all processes and deduplicates the entries based on unique_clip_id.

    Parameters:
    - gathered_results: List of dictionaries from all processes, gathered via torch.distributed.all_gather_object.
    - video_outputs: The main dictionary to update and deduplicate, keyed by unique_clip_id.

    Returns:
    - The updated and deduplicated video_outputs dictionary.
    """
    video_outputs = {}
    # Update video_outputs with gathered results and deduplicate
    for process_results in gathered_results:
        for clip_id, data in process_results.items():
            if clip_id not in video_outputs:
                video_outputs[clip_id] = data
            else:
                # If clip_id already exists, skip adding duplicates
                continue

    return video_outputs

def sliding_segment_with_mask(num_frames=48, num_segments=8):
    """
    Apply the VLM model using a sliding window approach within segments,
    ensuring all frames are covered. Also returns a mask for non-repeating elements.
    
    :param num_frames: total number of frames in the video
    :param num_segments: number of segments to divide the video into (default 8)
    :return: tuple (all_indices, masks)
        all_indices: list of lists, each containing frame indices for sampling
        masks: list of lists, with 1's for non-repeating elements and 0's for repeating
    """
    segment_size = num_frames // num_segments
    remainder = num_frames % num_segments
    all_indices = []
    masks = []

    # Track which frames have been seen
    seen_frames = set()

    # Handle the main part of the video
    for offset in range(segment_size):
        frame_indices = [i * segment_size + offset for i in range(num_segments)]
        mask = [1 if idx not in seen_frames else 0 for idx in frame_indices]
        all_indices.append(frame_indices)
        masks.append(mask)
        seen_frames.update(frame_indices)
    
    # Handle the remaining frames
    if remainder > 0:
        for offset in range(remainder):
            frame_indices = [i * segment_size + segment_size + offset for i in range(num_segments)]
            frame_indices = [idx for idx in frame_indices if idx < num_frames]
            if frame_indices:
                mask = [1 if idx not in seen_frames else 0 for idx in frame_indices]
                all_indices.append(frame_indices)
                masks.append(mask)
                seen_frames.update(frame_indices)

    return all_indices, masks

def inference(dataloader, model, tokenizer, token_embeddings, args, device):
    
    video_outputs = {}
    # Prepare inputs for inference
    input_ids = tokenize_prompt(tokenizer, args.use_mm_start_end)
    input_ids = input_ids.unsqueeze(0).cuda()
    # For mask decoder - avoiding recomputation as it is constant
    dense_pe = model(mode="get_dense_pe")
    for sample in tqdm(dataloader):
        global_enc_images_all = sample.global_enc_images.to(device)
        grounding_enc_images_all = sample.grounding_enc_images.to(device)
        original_sizes = sample.original_sizes
        clip_ids = sample.clip_ids
        video_ids = sample.video_ids

        # Sliding sparse sampling
        all_indices, masks = sliding_segment_with_mask(num_frames=global_enc_images_all.shape[2])
        # Use the centre frames of each segment to generate the caption
        for i, m in enumerate(masks):
            if all(m):
                last_index_all_ones = i
        center_indices = all_indices[last_index_all_ones // 2]
        all_indices_seen_order = center_indices

        global_enc_images = global_enc_images_all[:, :, center_indices]
        grounding_enc_images = grounding_enc_images_all[:, :, center_indices]

        image_features, image_forward_outs = model(images=global_enc_images, mode="encode_images")
        images_dtype = global_enc_images.dtype
        # Extract features from Grounding Image Encoder
        image_embeddings = model(images=grounding_enc_images, mode="get_grounding_encoder_embs")
        # Repeat tokens along batch dimension
        batch_input_ids = input_ids.repeat(global_enc_images.shape[0], 1).to(device=device)

        # Generate output
        batch_output_ids, pred_bboxes, logits_temp_objectness = model(mode='evaluate', 
                                                                      image_features=image_features, 
                                                                      image_forward_outs=image_forward_outs, 
                                                                      images_dtype=images_dtype, 
                                                                      image_embeddings=image_embeddings, 
                                                                      input_ids=batch_input_ids,
                                                                      original_size_list=original_sizes, 
                                                                      max_tokens_new=64, 
                                                                      bboxes=None, 
                                                                      token_embeddings=token_embeddings, 
                                                                      dense_pe=dense_pe, 
                                                                      device=device)
        batch_output_ids = [batch_output_ids[i][(batch_output_ids[i] != IMAGE_TOKEN_INDEX) & \
                                                    (batch_output_ids[i] != tokenizer.pad_token_id)]
                                                    for i in range(batch_output_ids.shape[0])]

        # Post-processing
        batch_text_output = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)
        for i in range(len(batch_output_ids)):
            text_output = batch_text_output[i].replace("\n", "").replace("  ", " ")
            text_output = text_output.split("ASSISTANT: ")[-1]

            cleaned_str = re.sub(r'<.*?>', '', text_output)

            pattern = re.compile(r'<p>(.*?)<\/p>')
            phrases = pattern.findall(text_output)
            phrases = [p.strip() for p in phrases]

            # Get phrases for each frame
            preds_temp_objectness_clip = [F.sigmoid(logits) > args.temp_objectness_threshold for logits in logits_temp_objectness[i]]
            preds_temp_objectness_clip = [preds.to(torch.float32).cpu().numpy() for preds in preds_temp_objectness_clip]
            phrases_per_frame = [[phrases[i] for i,p in enumerate(preds) if p and i < len(phrases)] for preds in preds_temp_objectness_clip]

            probs_temp_objectness_clip = [F.sigmoid(logits) for logits in logits_temp_objectness[i]]
            probs_temp_objectness_clip = [probs.to(torch.float32).cpu().numpy() for probs in probs_temp_objectness_clip]
            # Remove the [DET] token
            cleaned_str = cleaned_str.replace('[DET]', '')

            # Strip unnecessary spaces
            cleaned_str = ' '.join(cleaned_str.split()).strip("'")
            cleaned_str = cleaned_str.strip()

            # Clean caption tagged with [DET] but leave <p> </p> tags
            cleaned_tagged_str = text_output.replace('[DET]', '')
            cleaned_tagged_str = ' '.join(cleaned_tagged_str.split()).strip("'")
            cleaned_tagged_str = cleaned_tagged_str.strip()

            unique_clip_id = f"{video_ids[i]}_{clip_ids[i]}"
            if unique_clip_id not in video_outputs:
                video_outputs[unique_clip_id] = {'pred_bboxes': None, 'pred_labels': None, 'pred_phrases': None, 'pred_text': None, 'probs_temp_objectness': None}
            video_outputs[unique_clip_id]['pred_bboxes'] = [bbox.to(torch.float32).cpu().numpy() for bbox in pred_bboxes[i]]
            video_outputs[unique_clip_id]['pred_labels'] = phrases_per_frame
            video_outputs[unique_clip_id]['pred_phrases'] = phrases
            video_outputs[unique_clip_id]['pred_text'] = cleaned_str
            video_outputs[unique_clip_id]['pred_text_tagged'] = cleaned_tagged_str
            video_outputs[unique_clip_id]['probs_temp_objectness'] = probs_temp_objectness_clip

            # Given the caption of the center frames in each segment, get bbox predictions and temp objectness logits for each frame
            # of the remaining sliding windows

            for j, indices in enumerate(all_indices):
                if j == last_index_all_ones // 2:
                    continue

                global_enc_images = global_enc_images_all[:, :, indices]
                grounding_enc_images = grounding_enc_images_all[:, :, indices]

                input_ids_with_answer = tokenize_prompt(tokenizer, args.use_mm_start_end, answer=text_output)
                input_ids_with_answer = input_ids_with_answer.unsqueeze(0).to(device)

                preds = model(global_enc_images=global_enc_images, grounding_enc_images=grounding_enc_images,
                              bboxes_region=None, input_ids=input_ids_with_answer, labels=None,
                              attention_masks=None, offset=None, bboxes_list=None,
                              temp_objectness_labels_list=None, 
                              original_size_list=original_sizes, inference=True,)

                pred_bboxes, logits_temp_objectness = preds["pred_bboxes"], preds["logits_temp_objectness"]
                
                # Remove repeating elements using the mask
                pred_bboxes = [bbox for k, bbox in enumerate(pred_bboxes[0]) if masks[j][k]]
                logits_temp_objectness = [logits for k, logits in enumerate(logits_temp_objectness[0]) if masks[j][k]]
                indices = [idx for k, idx in enumerate(indices) if masks[j][k]]
                
                all_indices_seen_order.extend(indices)

                # Get phrases for each frame
                preds_temp_objectness_clip = [F.sigmoid(logits) > args.temp_objectness_threshold for logits in logits_temp_objectness]
                preds_temp_objectness_clip = [preds.to(torch.float32).cpu().numpy() for preds in preds_temp_objectness_clip]
                phrases_per_frame = [[phrases[i] for i,p in enumerate(preds) if p and i < len(phrases)] for preds in preds_temp_objectness_clip]

                probs_temp_objectness_clip = [F.sigmoid(logits) for logits in logits_temp_objectness]
                probs_temp_objectness_clip = [probs.to(torch.float32).cpu().numpy() for probs in probs_temp_objectness_clip]

                # Update video_outputs with the predictions for frames in the remaining sliding windows
                video_outputs[unique_clip_id]['pred_bboxes'].extend([bbox.to(torch.float32).cpu().numpy() for bbox in pred_bboxes])
                video_outputs[unique_clip_id]['pred_labels'].extend(phrases_per_frame)
                video_outputs[unique_clip_id]['probs_temp_objectness'].extend(probs_temp_objectness_clip)

            # Sort video outputs based on all_indices_seen_order_flattened
            sorted_indices = sorted(range(len(all_indices_seen_order)), key=lambda k: all_indices_seen_order[k])

            video_outputs[unique_clip_id]['pred_bboxes'] = [video_outputs[unique_clip_id]['pred_bboxes'][i] for i in sorted_indices]
            video_outputs[unique_clip_id]['pred_labels'] = [video_outputs[unique_clip_id]['pred_labels'][i] for i in sorted_indices]
            video_outputs[unique_clip_id]['probs_temp_objectness'] = [video_outputs[unique_clip_id]['probs_temp_objectness'][i] for i in sorted_indices]

    torch.distributed.barrier()
    video_output_list = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(video_output_list, video_outputs)
    all_video_outputs = update_and_sort_video_outputs(video_output_list)        

    return all_video_outputs


def init_transform_global_enc(transform_global_enc_config):
    return CLIPImageProcessor.from_pretrained(transform_global_enc_config)

def init_transform_grounding_enc(image_size):
    return ResizeLongestSide(image_size)

def grounding_enc_processor(x: torch.Tensor) -> torch.Tensor:
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 512

    # Normalization: normalize across the spatial dimensions but apply the same mean and std to all frames
    x = (x - IMG_MEAN[:, None, :, :]) / IMG_STD[:, None, :, :]
    
    # Padding: pad the spatial dimensions to match the desired IMG_SIZE
    # Assuming x has shape [C, T, H, W]
    _, _, h, w = x.shape
    padding = (0, IMG_SIZE - w, 0, IMG_SIZE - h)  # Right and bottom padding
    x = F.pad(x, padding)
    
    return x

class HowTo100M(torch.utils.data.Dataset):
    def __init__(
        self,
        video_dir,
        video_info,
        frame_size=512,
        target_fps=5,
        transform_global_enc_config="openai/clip-vit-large-patch14-336",
        temporal_sampling=True,
    ):
        """
        :param video_dir: str
        :param frame_info: str
        :param frame_size: int
        :param target_fps: int
        :param transform_global_enc_config: str
        """
        self.video_dir = video_dir
        self.video_info = pickle.load(open(video_info, "rb"))
        # Randomly select 100 key-values from self.video_info
        # subsampled_video_info = dict(random.sample(self.video_info.items(), 100))
        # Update self.video_info with the subsampled dict
        # self.video_info = subsampled_video_info
        self.video_clip_ids = list(self.video_info.keys())
        self.target_fps = target_fps
        self._transform_global_enc = init_transform_global_enc(transform_global_enc_config)
        self._transform_grounding_enc = init_transform_grounding_enc(frame_size)
        self.temporal_sampling = temporal_sampling

    def transform_global_enc(self, images_list):
        global_enc_images_list = self._transform_global_enc.preprocess(images_list, return_tensors="pt")["pixel_values"]
        global_enc_images_list = global_enc_images_list.bfloat16()
        global_enc_images_list = global_enc_images_list.permute(1, 0, 2, 3).contiguous()
        return global_enc_images_list
    
    def transform_grounding_enc(self, images_list):
        images_list = np.stack([self._transform_grounding_enc.apply_image(im) for im in images_list])
        grounding_enc_images_list = grounding_enc_processor(torch.from_numpy(images_list).permute(3, 0, 1, 2).contiguous())
        grounding_enc_images_list = grounding_enc_images_list.bfloat16()
        return grounding_enc_images_list

    def sample_frames(self, video_array, num_segments):
        """
        Sparse sampling of frames from a video array. In 'train' mode, it randomly
        samples a frame from each segment. In 'test' mode, it samples the center frame from
        each segment.
        
        :param video_array: A numpy array of shape (C, T, H, W).
        :param num_segments: The number of segments to divide the video into.
        :param mode: 'train' for training mode, 'test' for testing mode.
        :return: A numpy array of the sampled frames.
        """
        T = video_array.shape[1]
        segment_length = T // num_segments
        starts = np.arange(num_segments) * segment_length
        ends = np.append(starts[1:], T)

        # Middle points of each segment
        indices = (starts + ends) // 2

        return indices, video_array[:, indices, :, :]

    def __getitem__(self, idx):
        """
        :param idx: int
        :return:
        images: a CHW image tensor
        """
        video = self.video_info[self.video_clip_ids[idx]]
        # video = self.video_info[idx]
        video_id = video["video_id"]
        clip_id = video["clip_id"]
        start_frame = video["clip_start_frame"]
        end_frame = video["clip_end_frame"]
        w = video["width"]
        h = video["height"]
        video_path = os.path.join(self.video_dir, video["video_fname"])
        video_fps = video["fps"]
        
        # ffmpeg decoding
        sampling_rate = int(video_fps / self.target_fps)
        frame_ids = list(range(start_frame, end_frame + 1, sampling_rate)) # simulates the subsampling of frames during annotation
        
        all_frames = []
        for frame_id in frame_ids:
            timestamp = frame_id / video_fps
            try:
                out, _ = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
                all_frames.append(out)
            except ffmpeg.Error as e:
                print(e.stderr)
                raise e

        images_list = np.concatenate([np.frombuffer(frame, np.uint8).reshape([1, h, w, 3]) for frame in all_frames])
        assert len(images_list) == len(frame_ids), f"Length mismatch: {len(images_list)}, {len(frame_ids)}"
        
        global_enc_images_list = self.transform_global_enc(images_list)
        grounding_enc_images_list = self.transform_grounding_enc(images_list)

        original_size = (w, h)

        if not self.temporal_sampling:
            return global_enc_images_list, grounding_enc_images_list, original_size, clip_id, video_id

        temporal_indices, global_enc_images_list = self.sample_frames(global_enc_images_list, 8)
        grounding_enc_images_list = grounding_enc_images_list[:, temporal_indices]

        return global_enc_images_list, grounding_enc_images_list, original_size, clip_id, video_id
    
    def __len__(self) -> int:
        return len(self.video_info)
    

class CustomBatch:
    def __init__(self, batch):
        global_enc_images, grounding_enc_images, original_sizes, clip_ids, video_ids = zip(*batch)
        self.global_enc_images = torch.stack(global_enc_images, dim=0)
        self.grounding_enc_images = torch.stack(grounding_enc_images, dim=0)
        self.original_sizes = original_sizes
        self.clip_ids = clip_ids 
        self.video_ids = video_ids

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.global_enc_images = self.global_enc_images.pin_memory()
        self.grounding_enc_images = self.grounding_enc_images.pin_memory()
        return self

def collate_wrapper(batch):
    return CustomBatch(batch)

def initialize_model(args, tokenizer, rank):
    """ Initialize the GROVE model. """
    model_args = {k: getattr(args, k) for k in
                  ["det_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx", "num_frames", "temp_objectness_threshold"]}
    model_args["use_temp_objectness"] = True
    model = GROVEForCausalLM.from_pretrained(
        args.version, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", low_cpu_mem_usage=True, **model_args)
    if rank == 0:
        print('\033[92m' + "---- Initialized model from: {} ----".format(args.version) + '\033[0m')

    initialize_custom_layers_in_model(model)

    # Configure model tokens
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model

def prepare_model_for_inference(model, args, rank):
    # Initialize vision tower
    if rank == 0:
        print(
            '\033[92m' + "---- Initialized Global Image Encoder (vision tower) from: {} ----".format(
                args.vision_tower
            ) + '\033[0m'
        )
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()

    initialize_custom_layers_in_global_encoder(vision_tower)

    vision_tower.to(dtype=torch.bfloat16, device=rank)

    return model

if __name__ == "__main__":
    args = parse_args()

    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4.0))
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Initialize tokenizer and model
    tokenizer = setup_tokenizer_and_special_tokens(args)
    model = initialize_model(args, tokenizer, rank)
    model = prepare_model_for_inference(model, args, local_rank)
    interpolate_positional_embeddings(model)

    lora_r = args.lora_r
    if lora_r > 0:
        lora_config = setup_lora_config(model, args)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()

    # Print number of total parameters and trainable parameters
    model_parameters = model.parameters()
    params = sum(p.numel() for p in model_parameters)
    trainable_params = sum(p.numel() for p in model_parameters if p.requires_grad)
    print(f"Number of parameters: {params}\tTrainable parameters: {trainable_params}")


    state_dict = torch.load(args.grove_weights)
    if lora_r == 0:
        model.load_state_dict(state_dict, strict=False)
        print(f"Loading weights into GROVE from {args.grove_weights}.")
    else:
        updated_state_dict = {}
        for key in state_dict.keys():
            updated_key = f"base_model.model.{key}"
            updated_state_dict[updated_key] = state_dict[key]
        model.load_state_dict(updated_state_dict, strict=True)
        print(f"Successfully loaded weights into GROVE from {args.grove_weights}.")

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Create DDP Dataset
    dataset = HowTo100M(args.video_dir, args.video_info, temporal_sampling=False)
    sampler = torch.utils.data.DistributedSampler(
        dataset,
        shuffle=False,)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             sampler=sampler,
                                             batch_size=args.batch_size, 
                                             num_workers=args.num_workers, 
                                             collate_fn=collate_wrapper,
                                             pin_memory=True,
                                             multiprocessing_context="forkserver")

    token_embeddings = torch.load(args.token_embeddings)
    token_embeddings = token_embeddings.to(device=device)
    # Run inference
    with torch.no_grad():
        video_outputs = inference(dataloader, model, tokenizer, token_embeddings, args, device)

    # Save the inference results
    torch.distributed.barrier()
    if rank == 0:
        if not args.output_path.parent.exists():
            args.output_path.parent.mkdir(parents=True)
        with open(args.output_path, 'wb') as f:
            pickle.dump(video_outputs, f)
