from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
import ffmpeg
import re
import torch.nn.functional as F

from transformers import ProcessorMixin
from transformers import CLIPImageProcessor
from .model.llava import conversation as conversation_lib
import bleach
from .model.SAM.utils.transforms import ResizeLongestSide
from .model.llava.mm_utils import tokenizer_image_token
from pathlib import Path
from .tokenization_grove import GroveTokenizer
import json

from .utils.utils import (
    DEFAULT_VIDEO_TOKEN,
    DEFAULT_VID_START_TOKEN,
    DEFAULT_VID_END_TOKEN,
    IMAGE_TOKEN_INDEX,
)

DEFAULT_INSTRUCTION = (
    "Could you please give me a description of the video? Please respond with interleaved bounding boxes for the corresponding parts of the answer."
)


def select_center_frames(total_frames: int, num_desired: int) -> List[int]:
    """Sample_frames.
    Returns indices as list.
    """
    T = total_frames
    num_segments = num_desired
    segment_length = T // num_segments
    starts = np.arange(num_segments) * segment_length
    ends = np.append(starts[1:], T)
    indices = (starts + ends) // 2
    return [int(i) for i in indices]


def sam_normalize_and_pad(frames_chw: torch.Tensor, target_size: int = 512) -> torch.Tensor:
    """SAM grounding encoder preprocessing (normalize + pad).

    Args:
        frames_chw: Tensor (C, T, H, W) with pixel values 0..255 or 0..1.
        target_size: square side after padding (or resize if smaller dims exceed target).
    """
    IMG_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=frames_chw.dtype, device=frames_chw.device).view(-1, 1, 1)
    IMG_STD = torch.tensor([58.395, 57.12, 57.375], dtype=frames_chw.dtype, device=frames_chw.device).view(-1, 1, 1)
    x = frames_chw  # (C,T,H,W)
    x = (x - IMG_MEAN[:, None, :, :]) / IMG_STD[:, None, :, :]
    _, T, H, W = x.shape
    padding = (0, target_size - W, 0, target_size - H)  # (left, right, top, bottom)
    x = torch.nn.functional.pad(x, padding)
    return x


def resize_longest_side_frames(frames_bhwc: np.ndarray, resizer) -> np.ndarray:
    """Apply SAM ResizeLongestSide per frame."""
    return np.stack([resizer.apply_image(f) for f in frames_bhwc], axis=0)


def sliding_segment_with_mask(num_frames: int, num_segments: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Sliding window sampling with masks."""
    segment_size = num_frames // num_segments
    remainder = num_frames % num_segments
    all_indices: List[List[int]] = []
    masks: List[List[int]] = []
    seen = set()
    for offset in range(segment_size):
        frame_indices = [i * segment_size + offset for i in range(num_segments)]
        mask = [1 if idx not in seen else 0 for idx in frame_indices]
        all_indices.append(frame_indices)
        masks.append(mask)
        seen.update(frame_indices)
    if remainder > 0:
        for offset in range(remainder):
            frame_indices = [i * segment_size + segment_size + offset for i in range(num_segments)]
            frame_indices = [idx for idx in frame_indices if idx < num_frames]
            if frame_indices:
                mask = [1 if idx not in seen else 0 for idx in frame_indices]
                all_indices.append(frame_indices)
                masks.append(mask)
                seen.update(frame_indices)
    return all_indices, masks


@dataclass
class GroveOutput:
    input_ids: torch.Tensor
    global_enc_images: torch.Tensor
    grounding_enc_images: torch.Tensor
    original_size: Tuple[int, int]
    frame_indices: List[int]
    all_frame_indices: List[int]
    sliding_indices: Optional[List[List[int]]] = None
    sliding_masks: Optional[List[List[int]]] = None
    prompt: Optional[str] = None
    prompt_with_answer: Optional[str] = None
    answer_input_ids: Optional[torch.Tensor] = None


class GroveProcessor(ProcessorMixin):
    """Inference-only processor.

    Responsibilities:
      - Build prompt with optional video start/end wrapping.
      - Tokenize prompt (no answer text by default).
      - Decode / accept frames, sample center frames (num_frames).
      - Produce global encoder images using CLIPImageProcessor.
      - Produce grounding encoder images (SAM ResizeLongestSide + normalize/pad).

    Does NOT handle ground-truth labels or dataset-specific logic.
    """
    attributes = ["tokenizer"]

    def __init__(
        self,
        tokenizer,
        vision_tower: str = "openai/clip-vit-large-patch14-336",
        grounding_image_size: int = 512,
        num_frames: int = 8,
        use_mm_start_end: bool = True,
        target_fps: int = 5,
        conv_type: str = "llava_v1",
    ):
        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.use_mm_start_end = use_mm_start_end
        self._clip_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self._sam_resizer = ResizeLongestSide(grounding_image_size)
        self.grounding_image_size = grounding_image_size
        self.target_fps = target_fps
        self.conv_type = conv_type

    @classmethod
    def from_pretrained(cls, path_or_repo, **kwargs):
        tok = GroveTokenizer.from_pretrained(path_or_repo, **kwargs)

        cfg = {}
        p = Path(path_or_repo) / "preprocessor_config.json"
        if p.exists():
            cfg = json.loads(p.read_text())
            cfg.pop("processor_class", None)

        cfg.update(kwargs)
        return cls(tokenizer=tok, **cfg)

    def tokenize_prompt(self, instruction: Optional[str] = None, answer: str = "") -> str:
        """Construct conversation-formatted prompt.
        """
        conv = conversation_lib.conv_templates[self.conv_type].copy()
        conv.messages = []
        base_instruction = DEFAULT_INSTRUCTION if instruction is None else instruction
        instructions = bleach.clean(base_instruction)
        instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')
        instructions = instructions.replace('&lt;', '<').replace('&gt;', '>')  # double pass like legacy
        prompt_text = f"The {DEFAULT_VIDEO_TOKEN} provides an overview of the video.\n" + instructions
        if self.use_mm_start_end:
            replace_token = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN
            prompt_text = prompt_text.replace(DEFAULT_VIDEO_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt_text)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")

        return input_ids
    

    def process_frames(self, frames_bhwc: np.ndarray, temporal_sampling: bool) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """frames_bhwc: (T, H, W, 3) uint8 or float in [0,255]/[0,1].

        Args:
          temporal_sampling: If True select self.num_frames center frames; else keep all frames.
        """
        total = frames_bhwc.shape[0]
        if temporal_sampling:
            indices = select_center_frames(total, self.num_frames)
        else:
            indices = list(range(total))
        selected = frames_bhwc[indices]
        # Global encoder path
        clip_pixels = self._clip_processor.preprocess(list(selected), return_tensors="pt")["pixel_values"]  # (F,C,H,W)
        clip_pixels = clip_pixels.bfloat16().permute(1, 0, 2, 3).contiguous()  # (C,F,H,W)
        # Grounding encoder path
        resized = resize_longest_side_frames(selected, self._sam_resizer)  # (F,H',W',3)
        grounding = torch.from_numpy(resized).permute(3, 0, 1, 2).float().contiguous()  # (C,F,H',W')
        grounding = sam_normalize_and_pad(grounding, target_size=self.grounding_image_size).bfloat16()
        return clip_pixels, grounding, indices

    def decode_video(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
        video_fps: Optional[float] = None,
    ) -> np.ndarray:
        """Frame decoding with simple semantics:

        - If start_frame / end_frame provided: treat as explicit trim range (inclusive).
        - If omitted: infer full range via a single ffprobe (nb_frames if available else duration * fps).
        - Width / height / fps can be optionally provided; otherwise obtained from probe.
        """
        if start_frame is None or end_frame is None or w is None or h is None or video_fps is None:
            probe = ffmpeg.probe(video_path)
            vstream = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        if w is None:
            w = int(vstream['width'])
        if h is None:
            h = int(vstream['height'])
        if video_fps is None:
            fps_str = vstream.get('avg_frame_rate') or vstream.get('r_frame_rate') or '30/1'
            try:
                num, den = fps_str.split('/')
                video_fps = float(num) / float(den) if float(den) != 0 else float(num)
            except Exception:
                video_fps = 30.0
            if video_fps == 0:
                video_fps = 30.0
        if end_frame is None:
            nb_frames = vstream.get('nb_frames')
            if nb_frames and str(nb_frames).isdigit():
                total_frames = int(nb_frames)
            else:
                duration = vstream.get('duration')
                if duration is not None:
                    try:
                        total_frames = int(float(duration) * video_fps)
                    except Exception:
                        total_frames = int(video_fps * 60)
                else:
                    total_frames = int(video_fps * 60)
            end_frame = total_frames - 1
        sf = 0 if start_frame is None else start_frame
        ef = end_frame
        if ef < sf:
            raise ValueError("end frame must be >= start frame")
        sampling_rate = int(video_fps / self.target_fps) if video_fps > self.target_fps else 1
        frame_ids = list(range(sf, ef + 1, sampling_rate))
        frames = []
        for frame_id in frame_ids:
            timestamp = frame_id / video_fps
            try:
                out, _ = (
                    ffmpeg
                    .input(video_path, ss=timestamp)
                    .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                    .run(capture_stdout=True, quiet=True)
                )
                frame = np.frombuffer(out, np.uint8).reshape((h, w, 3))
                frames.append(frame)
            except ffmpeg.Error as e:
                print(e.stderr)
                raise e
        return np.stack(frames, axis=0)

    def __call__(
        self,
        video_path: str,
        instruction: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
        video_fps: Optional[float] = None,
        answer: Optional[str] = None,
        sliding_window: bool = False,
        temporal_sampling: bool = True,
    ) -> GroveOutput:
        """Run preprocessing with optional prior answer and sliding window metadata."""
        frames_bhwc = self.decode_video(video_path=video_path, start_frame=start_frame, end_frame=end_frame, w=w, h=h, video_fps=video_fps)
        if frames_bhwc.size == 0:
            raise ValueError(f"No frames decoded from {video_path}")
        original_size = (frames_bhwc.shape[2], frames_bhwc.shape[1])  # (W,H)
        if answer is not None:
            input_ids = self.tokenize_prompt(instruction, answer=None)
            answer_input_ids = self.tokenize_prompt(instruction, answer=answer)
        else:
            input_ids = self.tokenize_prompt(instruction, answer=None)
            answer_input_ids = None
        global_enc, grounding_enc, indices = self.process_frames(frames_bhwc, temporal_sampling=temporal_sampling)
        all_indices = list(range(frames_bhwc.shape[0]))
        sliding_indices = None
        sliding_masks = None
        if sliding_window:
            sliding_indices, sliding_masks = sliding_segment_with_mask(num_frames=len(all_indices), num_segments=self.num_frames)
        return GroveOutput(
            input_ids=input_ids,
            global_enc_images=global_enc,
            grounding_enc_images=grounding_enc,
            original_size=original_size,
            frame_indices=indices,
            all_frame_indices=all_indices,
            sliding_indices=sliding_indices,
            sliding_masks=sliding_masks,
            answer_input_ids=answer_input_ids,
        )

    @torch.no_grad()
    def generate(
        self,
        model,
        video_path: str,
        token_embeddings: torch.Tensor,
        device: torch.device,
        instruction: Optional[str] = None,
        max_new_tokens: int = 64,
        temp_objectness_threshold: Optional[float] = None,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        w: Optional[int] = None,
        h: Optional[int] = None,
        video_fps: Optional[float] = None,
    ) -> Dict[str, Any]:
        """High-level generation: caption + per-frame boxes with minimal duplication.

        Reuses __call__ to perform decoding, prompt building, and (here) disables temporal sampling to keep all frames.
        """
        # Ensure model has had special token IDs applied (from_pretrained handles this).
        if getattr(model, 'det_token_idx', None) is None:
            raise RuntimeError("Model det_token_idx is unset. Ensure model was loaded via GroveForCausalLM.from_pretrained which applies tokenizer special IDs.")
        prep = self(
            video_path=video_path,
            instruction=instruction or DEFAULT_INSTRUCTION,
            sliding_window=True,
            temporal_sampling=False,
            start_frame=start_frame,
            end_frame=end_frame,
            w=w,
            h=h,
            video_fps=video_fps,
        )
        clip_pixels = prep.global_enc_images.unsqueeze(0).to(device)  # (1,C,F,H,W) all frames
        grounding = prep.grounding_enc_images.unsqueeze(0).to(device)  # (1,C,F,H,W) all frames
        original_size = prep.original_size
        all_indices_windows = prep.sliding_indices
        masks = prep.sliding_masks
        if all_indices_windows is None or masks is None:
            raise RuntimeError("Sliding window metadata missing despite sliding_window=True in preparation.")
        last_index_all_ones = 0
        for i, m in enumerate(masks):
            if all(m):
                last_index_all_ones = i
        center_indices = all_indices_windows[last_index_all_ones // 2]
        global_center = clip_pixels[:, :, center_indices]
        grounding_center = grounding[:, :, center_indices]
        input_ids = prep.input_ids.unsqueeze(0).to(device)
        image_features, image_forward_outs = model(images=global_center, mode="encode_images")
        image_embeddings = model(images=grounding_center, mode="get_grounding_encoder_embs")
        dense_pe = model(mode="get_dense_pe")
        eval_out = model(mode='evaluate', image_features=image_features, image_forward_outs=image_forward_outs,
                         images_dtype=global_center.dtype, image_embeddings=image_embeddings, input_ids=input_ids,
                         original_size_list=[original_size], max_tokens_new=max_new_tokens, bboxes=None,
                         token_embeddings=token_embeddings, dense_pe=dense_pe, device=device)
        batch_output_ids, pred_bboxes_center, logits_temp_objectness_center = eval_out
        output_ids = batch_output_ids[0]
        output_ids = output_ids[(output_ids != IMAGE_TOKEN_INDEX) & (output_ids != self.tokenizer.pad_token_id)]
        raw_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = raw_text.replace("\n", "").replace("  ", " ")
        if "ASSISTANT: " in text_output:
            text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output).replace('[DET]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'").strip()
        cleaned_tagged_str = text_output.replace('[DET]', '')
        cleaned_tagged_str = ' '.join(cleaned_tagged_str.split()).strip("'").strip()
        phrase_pattern = re.compile(r'<p>(.*?)</p>')
        phrases = [p.strip() for p in phrase_pattern.findall(text_output)]
        threshold = temp_objectness_threshold if temp_objectness_threshold is not None else getattr(model.config, 'temp_objectness_threshold', 0.5)
        probs_temp_objectness_center = []
        phrases_per_frame_center = []
        if logits_temp_objectness_center is not None:
            for logits in logits_temp_objectness_center[0]:
                probs = torch.sigmoid(logits)
                mask = probs > threshold
                probs_temp_objectness_center.append(probs.to(torch.float32).cpu())
                phrases_per_frame_center.append([phrases[i] for i, p in enumerate(mask) if p and i < len(phrases)])
        else:
            phrases_per_frame_center = [[] for _ in center_indices]
        all_bboxes = [bbox.to(torch.float32).cpu() for bbox in pred_bboxes_center[0]]
        all_labels = phrases_per_frame_center
        all_probs = probs_temp_objectness_center
        all_indices_seen_order = list(center_indices)
        answer_input_ids = self.tokenize_prompt(instruction, answer=text_output).unsqueeze(0).to(device)
        for j, indices in enumerate(all_indices_windows):
            if indices == center_indices:
                continue
            mask = masks[j]
            global_subset = clip_pixels[:, :, indices]
            grounding_subset = grounding[:, :, indices]
            preds = model(global_enc_images=global_subset, grounding_enc_images=grounding_subset,
                          bboxes_region=None, input_ids=answer_input_ids, labels=None, attention_masks=None,
                          offset=None, bboxes_list=None, temp_objectness_labels_list=None,
                          original_size_list=[original_size], inference=True)
            pred_bboxes_window = preds['pred_bboxes']
            logits_temp_objectness_window = preds['logits_temp_objectness']
            filtered_indices = [idx for k, idx in enumerate(indices) if mask[k]]
            if not filtered_indices:
                continue
            filtered_bboxes = [bbox for k, bbox in enumerate(pred_bboxes_window[0]) if mask[k]]
            if logits_temp_objectness_window is not None:
                filtered_logits = [logits for k, logits in enumerate(logits_temp_objectness_window[0]) if mask[k]]
                for logits in filtered_logits:
                    probs = torch.sigmoid(logits)
                    all_probs.append(probs.to(torch.float32).cpu())
                    mask_thresh = probs > threshold
                    all_labels.append([phrases[i] for i, p in enumerate(mask_thresh) if p and i < len(phrases)])
            else:
                all_labels.extend([[] for _ in filtered_bboxes])
            all_bboxes.extend([b.to(torch.float32).cpu() for b in filtered_bboxes])
            all_indices_seen_order.extend(filtered_indices)
        sort_order = sorted(range(len(all_indices_seen_order)), key=lambda k: all_indices_seen_order[k])
        all_indices_sorted = [all_indices_seen_order[i] for i in sort_order]
        all_bboxes_sorted = [all_bboxes[i] for i in sort_order]
        all_labels_sorted = [all_labels[i] for i in sort_order]
        all_probs_sorted = [all_probs[i] for i in sort_order] if all_probs else []
        return {
            'text': cleaned_str,
            'text_tagged': cleaned_tagged_str,
            'phrases': phrases,
            'center_frame_indices': center_indices,
            'frame_indices': all_indices_sorted,
            'bboxes': all_bboxes_sorted,
            'labels_per_frame': all_labels_sorted,
            'probs_temp_objectness': all_probs_sorted,
            'original_size': original_size,
        }
        

__all__ = [
    'GroveProcessor',
    'GroveOutput'
]
