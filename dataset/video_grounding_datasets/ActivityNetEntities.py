import os
import re
import cv2
import json
import pickle
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import ffmpeg
from model.llava import conversation as conversation_lib
from model.SAM.utils.transforms import ResizeLongestSide
from utils.utils import DEFAULT_VIDEO_TOKEN
from utils.bbox_utils import normalize_bboxes, box_xyxy_to_cxcywh
from dataset.utils.utils import VIDEO_GROUNDING_QUESTIONS


class ActivityNetEntitiesDataset(torch.utils.data.Dataset):
    """
    Dataset Class for Video Grounding in ActivityNet-Entities dataset, operating at the segment level.
    """
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 512
    IGNORE_LABEL = 255

    def __init__(self, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=False,
                 fps=5, ann_path='', video_dir='', keys=None, frame_timestamps=None):
        """
        :param frame_timestamps: A dictionary where keys are video IDs and values are lists of frame timestamps.
        """
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
        self.validation = validation
        self.random_sampling = random_sampling
        self.fps = fps
        self.video_dir = video_dir
        self.frame_timestamps = frame_timestamps  # Added for variable FPS handling

        self.question_templates = VIDEO_GROUNDING_QUESTIONS
        self.begin_str = f"""The {DEFAULT_VIDEO_TOKEN} provides an overview of the video.\n"""

        with open(ann_path, "r") as f:
            self.annotations = json.load(f)['annotations']

        # Create a mapping of all segments
        self.segment_map = []
        for video_id in keys:
            video_data = self.annotations[video_id]
            segments = video_data.get("segments", {})
            for segment_id in segments:
                labels = [group[0].lower() for group in segments[segment_id]["process_clss"]]
                caption = " ".join(segments[segment_id]["tokens"]).lower()
                label_not_in_caption = not any(label in caption for label in labels)
                start_time, end_time = segments[segment_id]['timestamps']

                # Clamp end_time to frame_timestamps[-2]
                if video_id in self.frame_timestamps:
                    max_time = self.frame_timestamps[video_id][-2]
                    end_time = min(end_time, max_time)
 
                if (len(labels) != len(set(labels))
                    or not labels 
                    or label_not_in_caption 
                    or start_time >= end_time 
                    or (end_time - start_time) < 0.5):
                    continue
                self.segment_map.append((video_id, segment_id))

        mode = "Val" if validation else "Train"
        print('\033[92m' + f"----ActivityNetEntities-{mode}: Dataset initialized with {len(self.segment_map)} segments----" + '\033[0m')

    def _find_first_occurrences(self, caption, labels):
        """
        Find the first occurrence of each label in the caption, allowing for partial matches (e.g., "apple" matches "apples").
        """
        tokens_positive = []
        used_labels = set()  # To track labels that have already been tagged

        # Tokenize the caption to match against words
        words = caption.split()

        for label in labels:
            if label in used_labels:
                continue

            # Check each word in the caption for a match with the label
            for _, word in enumerate(words):
                if label in word:  # Check if the label is a substring of the word
                    # Find the character-level start and end positions
                    start = caption.find(word)
                    end = start + len(word)
                    tokens_positive.append((start, end))
                    used_labels.add(label)  # Mark this label as used
                    break  # Move to the next label after the first match

        return tokens_positive

    def _parse_annotations(self, video_id, segment_id):
        """
        Parse annotations for a specific segment.
        """
        video_data = self.annotations[video_id]
        fps = video_data["fps"]
        num_frames = video_data["num_frames"]
        duration = video_data["duration"]
        segment_data = video_data["segments"][segment_id]

        caption = " ".join(segment_data["tokens"]).lower()
        labels = [group[0].lower() for group in segment_data["process_clss"]]  # Use the first phrase only
        bboxes = [np.array(bbox) for bbox in segment_data["process_bnd_box"]]
        frame_indices = segment_data["frame_ind"]
        timestamps = segment_data["timestamps"]

        # Find positions of phrases in the caption
        tokens_positive = self._find_first_occurrences(caption, labels)

        # Sort by the appearance of the phrases in the caption
        sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
        tokens_positive = [tokens_positive[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        bboxes = [bboxes[i] for i in sorted_indices]
        frame_indices = [frame_indices[i] for i in sorted_indices]

        youtube_video_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
        for ext in youtube_video_extensions:
            video_path = os.path.join(self.video_dir, f"{video_id}{ext}")
            if os.path.exists(video_path):
                break

        return {
            'video_path': video_path,
            'caption': caption,
            'labels': labels,
            'bboxes': bboxes,
            'tokens_positive': tokens_positive,
            'frame_indices': frame_indices,
            'timestamps': timestamps,
            'fps': fps,
            'num_frames': num_frames,
            'duration': duration
        }

    def sample_timestamps(self, start_time, end_time, num_segments, annotated_timestamps):
        """
        Sample timestamps while ensuring all annotated timestamps are included.

        :param start_time: Start time of the segment.
        :param end_time: End time of the segment.
        :param num_segments: Number of segments for sampling.
        :param annotated_timestamps: List of annotated timestamps to include.
        :return: List of sampled timestamps.
        """
        segment_boundaries = np.linspace(start_time, end_time, num_segments + 1)
        sampled_timestamps = []

        for i in range(num_segments):
            segment_start = segment_boundaries[i]
            segment_end = segment_boundaries[i + 1]
            
            # Check if any annotated timestamp falls in this segment
            annotated_in_segment = [t for t in annotated_timestamps if segment_start <= t < segment_end]
            if annotated_in_segment:
                if not self.validation:
                    # Randomly pick one annotated timestamp during training
                    sampled_timestamps.append(np.random.choice(annotated_in_segment))
                else:
                    # Always pick the first annotated timestamp during validation
                    sampled_timestamps.append(annotated_in_segment[0])
            elif not self.validation:
                # Randomly sample within the segment
                sampled_timestamps.append(np.random.uniform(segment_start, segment_end))
            else:
                # Sample the midpoint for validation
                sampled_timestamps.append((segment_start + segment_end) / 2)

        return sampled_timestamps

    def _tag_caption(self, caption, tokens_positive):
        """
        Add <p> tags around the phrases in the caption.
        """
        for start, end in sorted(tokens_positive, key=lambda x: x[0], reverse=True):
            caption = f"{caption[:start]}<p> {caption[start:end]} </p> [DET]{caption[end:]}"
        return caption

    def create_conversations(self, caption, tokens_positive):
        """
        Generate question-answer pairs and conversations.
        """
        question = random.choice(self.question_templates).strip()

        # Prepare tagged caption
        detailed_answer = self._tag_caption(caption, tokens_positive)

        # Prepare conversation
        conversations = []
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], detailed_answer)
        conversations.append(conv.get_prompt())
        questions = [question]

        return questions, conversations

    def compute_midpoint_timestamps(self, start_time, end_time, total_segments=10):
        """
        Compute the midpoint timestamps of equally spaced segments between start_time and end_time.

        :param start_time: Start time of the video segment (in seconds).
        :param end_time: End time of the video segment (in seconds).
        :param total_segments: Number of temporal segments (default 10).
        :return: List of midpoint timestamps.
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
                ffmpeg
                .input(video_path, ss=adjusted_timestamp)
                .filter('scale', 720, -1)  # Add the scaling filter
                .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
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

    def __getitem__(self, index):
        video_id, segment_id = self.segment_map[index]
        annotations = self._parse_annotations(video_id, segment_id)

        bboxes = annotations["bboxes"]
        labels = annotations["labels"]
        tokens_positive = annotations["tokens_positive"]
        caption = annotations["caption"]
        frame_indices = annotations["frame_indices"]
        start_time, end_time = annotations["timestamps"]
        youtube_video_extensions = [".mp4", ".mov", ".mkv", ".avi", ".webm"]
        for ext in youtube_video_extensions:
            video_path = os.path.join(self.video_dir, f"{video_id}{ext}")
            if os.path.exists(video_path):
                break

        # Clamp end_time to frame_timestamps[-2]
        frame_timestamps = self.frame_timestamps[video_id]
        end_time = min(end_time, frame_timestamps[-2])

        # Compute the 10 midpoint timestamps
        midpoint_timestamps = self.compute_midpoint_timestamps(start_time, end_time, total_segments=10)

        # Annotated timestamps
        annotated_timestamps = [midpoint_timestamps[idx] for idx in frame_indices]

        # Sample timestamps using sample_timestamps
        sampled_timestamps = self.sample_timestamps(start_time, end_time, 8, annotated_timestamps)

        # Load video frames at sampled timestamps
        images_list, w, h = self._load_video(video_path, sampled_timestamps, frame_timestamps)

        # Prepare temp_objectness_labels
        temp_objectness_labels = np.zeros((len(sampled_timestamps), len(labels)))
        bboxes_all_frames = [[] for _ in range(len(sampled_timestamps))]
        for label_idx, annotated_time in zip(range(len(labels)), annotated_timestamps):
            if annotated_time in sampled_timestamps:
                local_idx = sampled_timestamps.index(annotated_time)
                temp_objectness_labels[local_idx, label_idx] = 1
                bboxes_all_frames[local_idx].append(bboxes[label_idx])

        # Normalize bounding boxes
        bboxes_all_frames = [np.array(frame_bboxes) for frame_bboxes in bboxes_all_frames]
        bboxes_all_frames = [box_xyxy_to_cxcywh(frame_bboxes) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes_all_frames]
        bboxes_all_frames = [normalize_bboxes(frame_bboxes, w, h) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes_all_frames]
        bboxes_all_frames = [torch.from_numpy(frame_bboxes).float() for frame_bboxes in bboxes_all_frames]

        # Convert temp_objectness_labels to tensors
        temp_objectness_labels = [torch.from_numpy(temp_objectness_labels[i]) for i in range(len(temp_objectness_labels))]

        # Prepare inputs for encoders
        global_enc_images_list_sampled = self.global_enc_processor.preprocess(images_list, return_tensors="pt")[
            "pixel_values"
        ].permute(1, 0, 2, 3).contiguous()

        images_list = np.stack([self.transform.apply_image(im) for im in images_list])
        grounding_enc_images_list_sampled = self.grounding_enc_processor(
            torch.from_numpy(images_list).permute(3, 0, 1, 2).contiguous()
        )

        questions, conversations = self.create_conversations(caption, tokens_positive)

        return (
            video_path,
            global_enc_images_list_sampled,
            grounding_enc_images_list_sampled,
            None,
            conversations,
            bboxes_all_frames,
            (w, h),
            questions,
            labels,
            temp_objectness_labels,
        )

    def __len__(self):
        return len(self.segment_map)

    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN[:, None, :, :]) / self.IMG_STD[:, None, :, :]
        _, _, h, w = x.shape
        padding = (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h)
        x = F.pad(x, padding)
        return x