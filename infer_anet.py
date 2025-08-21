import os
import re
import cv2
import json
import pickle
import bleach
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
    parser = argparse.ArgumentParser(description="GROVE Inference - ActivityNetEntities")

    parser.add_argument("--version", default="MBZUAI/GLaMM-GranD-Pretrained")
    parser.add_argument("--grove_weights", default="/home/grove_checkpoints/grove_ft_anet_ckpt.bin", type=str)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--vision-tower", default="openai/clip-vit-large-patch14-336", type=str)
    parser.add_argument("--video_dir", default="/home/ActivityNetEntities/videos/", type=str)
    parser.add_argument("--frame_timestamps", default="/home/ActivityNetEntities_annotations/timestamps_metadata.json")
    parser.add_argument("--ann_path", default="/home/ActivityNetEntities_annotations/data/", type=str)
    parser.add_argument("--output_path", type=Path, default=Path("/home/grove_inference_output/result_anet.pkl"))
    parser.add_argument("--token_embeddings", default="/home/token_embeddings_video.pt", type=str)
    parser.add_argument("--image_size", default=512, type=int, help="image size")
    parser.add_argument("--num_frames", default=8, type=int)
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
    Updates the video_outputs dict with gathered results from all processes and deduplicates entries.
    
    Parameters:
    - gathered_results: List of dictionaries from all processes, gathered via torch.distributed.all_gather_object.
    
    Returns:
    - The updated and deduplicated video_outputs dictionary.
    """
    video_outputs = {}
    # Update video_outputs with gathered results and deduplicate
    for process_results in gathered_results:
        for video_id, segments in process_results.items():
            if video_id not in video_outputs:
                video_outputs[video_id] = {}
            for segment_id, data in segments.items():
                if segment_id not in video_outputs[video_id]:
                    video_outputs[video_id][segment_id] = data
                else:
                    # If segment_id already exists, skip adding duplicates
                    continue

    return video_outputs

def sliding_segments(num_segments=10, window_size=8):
    """
    Generate three sets of frame indices for sliding windows:
    - Set 1: Segments 0-7
    - Set 2: Segments 1-8
    - Set 3: Segments 2-9
    """
    assert window_size <= num_segments, "Window size cannot exceed the total number of segments"
    return [
        list(range(start, start + window_size)) for start in range(num_segments - window_size + 1)
    ]


def inference(dataloader, model, tokenizer, token_embeddings, args, device):
    results = {
        "results": {},
        "eval_mode": "gen",
        "external_data": {
            "used": True,
            "details": "Object detector pre-trained on Visual Genome on object detection task.",
        },
    }

    video_outputs = {}

    # Sliding windows: segments [0-7], [1-8], [2-9]
    sliding_windows = sliding_segments(num_segments=10, window_size=8)

    # Prepare inputs for inference
    input_ids = tokenize_prompt(tokenizer, args.use_mm_start_end)
    input_ids = input_ids.unsqueeze(0).to(device)
    dense_pe = model(mode="get_dense_pe")

    for sample in tqdm(dataloader):
        # Access attributes from CustomBatch
        global_enc_images_all = sample.global_enc_images.to(device)
        grounding_enc_images_all = sample.grounding_enc_images.to(device)
        original_sizes = sample.original_sizes
        video_ids = sample.video_ids
        segment_ids = sample.segment_ids
        frame_indices = sample.frame_indices
        start_time = sample.start_time
        end_time = sample.end_time

        # First pass: Generate caption and predict bounding boxes for frames 0-7
        primary_window = sliding_windows[0]
        global_enc_images = global_enc_images_all[:, :, primary_window]
        grounding_enc_images = grounding_enc_images_all[:, :, primary_window]

        image_features, image_forward_outs = model(images=global_enc_images, mode="encode_images")
        images_dtype = global_enc_images_all.dtype
        image_embeddings = model(images=grounding_enc_images, mode="get_grounding_encoder_embs")

        batch_input_ids = input_ids.repeat(global_enc_images.shape[0], 1).to(device)

        # Generate caption and bounding boxes
        batch_output_ids, pred_bboxes = model(
            mode="evaluate",
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
            device=device,
        )
        # Process output
        batch_output_ids = [
            batch_output_ids[i][
                (batch_output_ids[i] != IMAGE_TOKEN_INDEX)
                & (batch_output_ids[i] != tokenizer.pad_token_id)
            ]
            for i in range(batch_output_ids.shape[0])
        ]
        batch_text_output = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=False)

        for i, text_output in enumerate(batch_text_output):
            text_output = text_output.replace("\n", "").replace("  ", " ").strip()
            text_output = text_output.split("ASSISTANT: ")[-1]
            if text_output == "<p> he </p> [DET] then covers the <p> window </p> [DET] with <p> paper </p> [DET] and puts up the <p> wall </p> [DET] aper </p> [DET] </s>":
                text_output = "<p> he </p> [DET] then covers the <p> window </p> [DET] with <p> paper </p> [DET] and puts up the <p> wall </p> [DET] <p> paper </p> [DET] </s>"
            if text_output == "a <p> screen </p> [DET] with a black <p> background </p> [DET] appears with white <p> words </p> [DET] that read here it is stylist </p> [DET] stylist stylist stylist stylist stylist stylist stylist stylist stylist stylist stylist stylist":
                text_output = "a <p> screen </p> [DET] with a black <p> background </p> [DET] appears with white <p> words </p> [DET] that read <p> here it is stylist </p> [DET] </s>"
            if text_output == "<p> she </p> [DET] puts on a black <p> hat </p> [DET] and puts on a black <p> lip </p> [DET] stencil </p> [DET] </s>":
                text_output = "<p> she </p> [DET] puts on a black <p> hat </p> [DET] and puts on a black <p> lip </p> [DET] <p> stencil </p> [DET] </s>"
            if text_output == "the <p> man </p> [DET] then begins to apply the <p> wallp </p> [DET] aper </p> [DET] </s>":
                text_output = "the <p> man </p> [DET] then begins to apply the <p> wall </p> [DET] <p> paper </p> [DET] </s>"
            if text_output == "<p> he </p> [DET] is shown several times riding around a <p> skate </p> [DET] [DET] park</s>":
                text_output = "<p> he </p> [DET] is shown several times riding around a <p> skate </p> [DET] <p> park </p> [DET] </s>"
            if text_output == "a <p> woman </p> [DET] is hanging <p> wallp </p> [DET] aper </p> [DET] on a wall</s>":
                text_output = "a <p> woman </p> [DET] is hanging <p> wall </p> [DET] <p> paper </p> [DET] on a wall</s>"
            if text_output == "the <p> words </p> [DET]  [DET] <p> logo </p> [DET] appear on the <p> screen </p> [DET] </s>":
                text_output = "the <p> words </p> [DET] <p> object </p> [DET] <p> logo </p> [DET] appear on the <p> screen </p> [DET] </s>"
            if text_output == "a <p> <p> </p> [DET] <p> <p> </p> [DET] <p> [DET] <p> </p> [DET] <p> [DET] <p> [DET] <p> [DET] </p> [DET] <p> [DET] <p> [DET] </p> [DET] <p> [DET] <p> [DET]":
                text_output = "a <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET] <p> object </p> [DET]"
            if text_output == "the <p> man </p> [DET] in the red <p> shirt </p> [DET] tries to stop him , but the <p> person </p> [DET] in black </p> [DET] jumps over him</s>":
                text_output = "the <p> man </p> [DET] in the red <p> shirt </p> [DET] tries to stop him , but the <p> person </p> [DET] in <p> black </p> [DET] jumps over him</s>"
            if text_output == "<p> he </p> [DET] is using a <p> leaf </p> [DET] blower </p> [DET] to blow leaves</s>":
                text_output = "<p> he </p> [DET] is using a <p> leaf </p> [DET] <p> blower </p> [DET] to blow leaves</s>"
            cleaned_str = re.sub(r"<.*?>", "", text_output)

            # Extract phrases from caption
            pattern = re.compile(r"<p>(.*?)<\/p>")
            phrases = pattern.findall(text_output)
            phrases = [p.strip() for p in phrases]

            # Remove the [DET] token if it exists
            cleaned_str = cleaned_str.replace("[DET]", "")
            # Strip unnecessary spaces
            cleaned_str = " ".join(cleaned_str.split()).strip("'").strip()

            # Initialize bounding box results for all 10 segments
            bbox_for_all_frames = [[] for _ in range(len(phrases))]
            # Assign bounding boxes for frames in the primary window (0-7)
            for j, segment_idx in enumerate(primary_window):
                for cls_idx, bbox in enumerate(pred_bboxes[i][j]):
                    try:
                        if len(bbox_for_all_frames[cls_idx]) <= segment_idx:
                            bbox_for_all_frames[cls_idx].append(bbox.float().cpu().numpy().tolist())
                    except IndexError:
                        print(f"IndexError: cls_idx={cls_idx}, bbox_for_all_frames={bbox_for_all_frames}, pred_bboxes={pred_bboxes}, phrases={phrases}, text_output={text_output}")
                        raise IndexError
            # Second and third passes: Predict bounding boxes for frames 8 and 9
            for window in sliding_windows[1:]:
                global_enc_images = global_enc_images_all[:, :, window]
                grounding_enc_images = grounding_enc_images_all[:, :, window]

                input_ids_with_answer = tokenize_prompt(tokenizer, args.use_mm_start_end, answer=text_output)
                input_ids_with_answer = input_ids_with_answer.unsqueeze(0).to(device)

                preds = model(
                    global_enc_images=global_enc_images, 
                    grounding_enc_images=grounding_enc_images,
                    bboxes_region=None,
                    input_ids=input_ids_with_answer,
                    labels=None,
                    attention_masks=None, 
                    offset=None, 
                    bboxes_list=None,
                    temp_objectness_labels_list=None, 
                    original_size_list=original_sizes,
                    inference=True,
                )
                pred_bboxes = preds["pred_bboxes"]

                # Assign bounding boxes for frame 8 (for second window) and frame 9 (for third window)
                frame_to_assign = window[-1]
                for cls_idx, bbox in enumerate(pred_bboxes[0][-1]):
                    if len(bbox_for_all_frames[cls_idx]) <= frame_to_assign:
                        bbox_for_all_frames[cls_idx].append(bbox.float().cpu().numpy().tolist())

            # Generate indices of phrases in the caption
            idx_in_sent = [cleaned_str.find(phrase) for phrase in phrases]

            # Store results for this sample
            if video_ids[i] not in video_outputs:
                video_outputs[video_ids[i]] = {}
            video_outputs[video_ids[i]][segment_ids[i]] = {
                "clss": phrases,
                "idx_in_sent": idx_in_sent,
                "bbox_for_all_frames": bbox_for_all_frames,
            }

    # Synchronize and gather results across processes for distributed inference
    torch.distributed.barrier()
    video_output_list = [None for _ in range(torch.distributed.get_world_size())]
    dist.all_gather_object(video_output_list, video_outputs)

    # Update and deduplicate results
    deduplicated_video_outputs = update_and_sort_video_outputs(video_output_list)

    # Merge deduplicated results into final results format
    for video_id, segments in deduplicated_video_outputs.items():
        if video_id not in results["results"]:
            results["results"][video_id] = {}
        results["results"][video_id].update(segments)

    return results


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


class ActivityNetEntitiesInferenceDataset(torch.utils.data.Dataset):
    """
    Inference Dataloader for ActivityNet-Entities dataset.
    """

    def __init__(
        self,
        video_dir,
        ann_path,
        frame_timestamps,
        keys=None,
        frame_size=512,
        transform_global_enc_config="openai/clip-vit-large-patch14-336",
    ):
        """
        :param video_dir: Path to video directory.
        :param ann_path: Path to the annotation file.
        :param frame_timestamps: Dictionary mapping video IDs to lists of frame timestamps.
        :param frame_size: Size of frames for the grounding encoder.
        :param transform_global_enc_config: Configuration for the global encoder.
        """
        self.video_dir = video_dir
        self.frame_timestamps = frame_timestamps
        self._transform_global_enc = init_transform_global_enc(transform_global_enc_config)
        self._transform_grounding_enc = init_transform_grounding_enc(frame_size)

        with open(ann_path, "r") as f:
            self.annotations = json.load(f)["annotations"]

        # Create a mapping of all segments
        self.segment_map = []
        for video_id in keys:
            video_data = self.annotations[video_id]
            segments = video_data.get("segments", {})
            for segment_id in segments:
                self.segment_map.append((video_id, segment_id))
    
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

    def compute_midpoint_timestamps(self, start_time, end_time, total_segments=10):
        """
        Compute the midpoint timestamps of equally spaced segments between start_time and end_time.
        """
        segment_boundaries = np.linspace(start_time, end_time, total_segments + 1)
        midpoints = [(segment_boundaries[i] + segment_boundaries[i + 1]) / 2 for i in range(total_segments)]
        return midpoints

    def _load_video(self, video_path, timestamps, frame_timestamps):
        """
        Extract frames at specific timestamps.

        :param video_path: Path to the video file.
        :param timestamps: List of timestamps (in seconds) at which to extract frames.
        :param frame_timestamps: List of extracted frame timestamps for the video.
        :return: images_list, w, h
        """
        all_frames = []
        for timestamp in timestamps:
            # Find the closest available frame timestamp
            closest_frame_idx = np.argmin([abs(ft - timestamp) for ft in frame_timestamps])
            adjusted_timestamp = frame_timestamps[closest_frame_idx]

            out, _ = (
                ffmpeg.input(video_path, ss=adjusted_timestamp)
                .filter("scale", 720, -1)
                .output("pipe:", vframes=1, format="rawvideo", pix_fmt="rgb24")
                .run(capture_stdout=True, quiet=True)
            )
            all_frames.append(out)

        # Extract width and height dynamically from the first frame
        first_frame = np.frombuffer(all_frames[0], np.uint8)
        w = 720
        h = first_frame.shape[0] // (w * 3)

        # Prepare images list
        images_list = np.concatenate([np.frombuffer(frame, np.uint8).reshape([1, h, w, 3]) for frame in all_frames])

        return images_list, w, h

    def __getitem__(self, idx):
        """
        :param idx: Index of the segment to process.
        :return: Processed data for inference.
        """
        video_id, segment_id = self.segment_map[idx]
        video_data = self.annotations[video_id]
        segment_data = video_data["segments"][segment_id]

        frame_indices = segment_data["frame_ind"]
        start_time, end_time = segment_data["timestamps"]
        youtube_video_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
        for ext in youtube_video_extensions:
            video_path = os.path.join(self.video_dir, f"{video_id}{ext}")
            if os.path.exists(video_path):
                break
        frame_timestamps = self.frame_timestamps[video_id]

        # Clamp end_time to frame_timestamps[-2]
        end_time = min(end_time, frame_timestamps[-2])

        # Compute the 10 midpoint timestamps
        midpoint_timestamps = self.compute_midpoint_timestamps(start_time, end_time, total_segments=10)

        # Load video frames at sampled timestamps
        images_list, w, h = self._load_video(video_path, midpoint_timestamps, frame_timestamps)

        global_enc_images_list = self.transform_global_enc(images_list)
        grounding_enc_images_list = self.transform_grounding_enc(images_list)

        return (
            global_enc_images_list,
            grounding_enc_images_list,
            (w, h),
            video_id,
            segment_id,
            frame_indices,
            start_time,
            end_time
        )

    def __len__(self):
        return len(self.segment_map)


class CustomBatch:
    def __init__(self, batch):
        (
            global_enc_images,
            grounding_enc_images,
            original_sizes,
            video_ids,
            segment_ids,
            frame_indices,
            start_time,
            end_time
        ) = zip(*batch)

        self.global_enc_images = torch.stack(global_enc_images, dim=0)
        self.grounding_enc_images = torch.stack(grounding_enc_images, dim=0)
        self.original_sizes = original_sizes
        self.video_ids = video_ids
        self.segment_ids = segment_ids
        self.frame_indices = frame_indices
        self.start_time = start_time
        self.end_time = end_time

    # Custom memory pinning method on custom type
    def pin_memory(self):
        self.global_enc_images = self.global_enc_images.pin_memory()
        self.grounding_enc_images = self.grounding_enc_images.pin_memory()
        return self

def collate_wrapper(batch):
    return CustomBatch(batch)

def initialize_model(args, tokenizer, rank):
    """ Initialize the GROVE model. """
    model_args = {k: getattr(args, k) for k in
                  ["det_token_idx", "bbox_token_idx", "eop_token_idx", "bop_token_idx", "num_frames"]}
    model_args["use_temp_objectness"] = False
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

    dist.init_process_group("nccl")
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
        model.load_state_dict(state_dict, strict=True)
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
    frame_timestamps = json.load(open(args.frame_timestamps, 'r'))
    val_keys = json.load(open(os.path.join(args.ann_path, 'split_ids_anet_entities.json'), 'r'))
    val_keys = val_keys["validation"]
    ann_path = os.path.join(args.ann_path, 'anet_entities_cleaned_class_thresh50_trainval.json')
    dataset = ActivityNetEntitiesInferenceDataset(args.video_dir, 
                                                  ann_path,
                                                  frame_timestamps,
                                                  keys=val_keys)
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
        with open(args.output_path, 'w') as f:
            json.dump(video_outputs, f)
