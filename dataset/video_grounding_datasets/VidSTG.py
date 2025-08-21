import os
import numpy as np
import random
import json
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor
import ffmpeg
from utils.bbox_utils import normalize_bboxes, box_xyxy_to_cxcywh
from model.SAM.utils.transforms import ResizeLongestSide
from model.llava import conversation as conversation_lib
from utils.utils import DEFAULT_VIDEO_TOKEN
from dataset.utils.utils import VIDEO_STG_QUESTIONS

class VidSTGDataset(torch.utils.data.Dataset):
    """
    Spatio-Temporal Grounding Dataset for VidSTG task.
    """
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 512
    IGNORE_LABEL = 255

    def __init__(self, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=False,
                 fps=5, ann_path='', video_dir=''):
        """
        :param ann_file: Path to the annotation file.
        :param video_dir: Path to the directory containing video files.
        :param tokenizer: Tokenizer to process captions.
        :param validation: Flag to indicate validation mode.
        :param fps: Desired FPS for subsampling video frames.
        """
        self.epoch_samples = epoch_samples
        self.num_classes_per_sample = num_classes_per_sample
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.annotations = self._load_annotations(ann_path)
        self.video_dir = video_dir
        self.validation = validation
        self.random_sampling = random_sampling
        self.fps = fps

        self.transform = ResizeLongestSide(image_size)
        self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)

        # Question template and initialization
        self.question_templates = VIDEO_STG_QUESTIONS
        self.begin_str = f"""The {DEFAULT_VIDEO_TOKEN} provides an overview of the video.\n"""

        mode = "Val" if validation else "Train"
        print('\033[92m' + f"----VidSTG-{mode}: Dataset initialized with {len(self.annotations['videos'])} segments----" + '\033[0m')

    def _load_annotations(self, ann_file):
        """
        Load annotations from the JSON file.
        """
        with open(ann_file, "r") as f:
            data = json.load(f)
        return data

    def _load_video(self, video_path, frame_ids, h, w, video_fps):
        """
        Load video frames at a specified FPS between start and end frames.

        :param video_path: Path to the video file.
        :param start_frame: Start frame index.
        :param end_frame: End frame index.
        :param video_fps: Original FPS of the video.
        :return: List of frames.
        """
        all_frames = []
        for frame_id in frame_ids:
            timestamp = frame_id / video_fps
            out, _ = (
                ffmpeg
                .input(video_path, ss=timestamp)
                .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
                .run(capture_stdout=True, quiet=True)
            )
            all_frames.append(out)

        images_list = np.concatenate([np.frombuffer(frame, np.uint8).reshape([1, h, w, 3]) for frame in all_frames])

        return images_list

    def sample_frames(self, total_frames, num_segments=8):
        """
        Sample frames from the video based on the number of segments.

        :param total_frames: Total number of frames in the video.
        :return: Indices of sampled frames.
        """
        if total_frames == num_segments:
            # Use all frames directly
            return np.arange(total_frames)
        
        elif total_frames < num_segments:
            # Use all frames and pad with the last frame
            padded_indices = np.pad(
                np.arange(total_frames),
                (0, num_segments - total_frames),
                mode='edge'
            )
            return padded_indices
        
        segment_length = total_frames // num_segments
        starts = np.arange(num_segments) * segment_length
        ends = np.append(starts[1:], total_frames)

        if not self.validation:
            indices = [np.random.randint(start, end) for start, end in zip(starts, ends)]
        else:
            indices = [(start + end) // 2 for start, end in zip(starts, ends)]

        return np.array(indices)

    def create_conversations(self, caption):
        """
        Generate question-answer pairs and conversations.
        """
        question = random.choice(self.question_templates).strip()
        detailed_answer = f"<p> {caption.strip()} </p> [DET]"  # Use tagged caption

        # Prepare conversation
        conversations = []
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], detailed_answer)
        conversations.append(conv.get_prompt())
        questions = [question]

        return questions, conversations
    
    def process_bboxes(self, w, h, anno):
        """
        :param w: pixel width of the frame
        :param h: pixel height of the frame
        :param anno: dictionary with key 'bbox'
        :return: List of processed bounding boxes in [x_min, y_min, x_max, y_max] format.
        """
        boxes = [obj["bbox"] for obj in anno]
        # Guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Filter invalid boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        # Convert to list of lists for output
        return boxes.tolist()

    def __getitem__(self, idx):
        """
        Get a single sample.

        :param idx: Index of the sample.
        :return: Processed video frames, caption, bounding boxes, and temporal information.
        """
        video_info = self.annotations["videos"][idx]
        original_video_id = video_info["original_video_id"]
        video_path = os.path.join(self.video_dir, video_info["video_path"])
        caption = video_info["caption"]
        tube_start_frame = video_info["tube_start_frame"]
        tube_end_frame = video_info["tube_end_frame"]
        video_fps = video_info["fps"]
        bounding_boxes = self.annotations["trajectories"][original_video_id][str(video_info["target_id"])]
        w = video_info["width"]
        h = video_info["height"]

        # Subsample frames at self.fps
        sampling_rate = int(video_fps / self.fps)
        all_frame_ids = list(range(tube_start_frame, tube_end_frame - 1, sampling_rate))

        # Sparse sampling of frames using 8 segments
        sampled_indices = self.sample_frames(len(all_frame_ids))
        sampled_frame_ids = [all_frame_ids[idx] for idx in sampled_indices]

        # Load video and sampled frames
        images_list = self._load_video(video_path, sampled_frame_ids, h, w, video_fps)

        temp_objectness_labels = [
            torch.ones(1, dtype=torch.float32) for _ in range(len(sampled_indices))
        ]

        # Sample bounding boxes for the selected frames
        bboxes_all_frames = []
        for frame_idx in sampled_frame_ids:
            if str(frame_idx) in bounding_boxes:
                # Prepare bounding boxes using the prepare() function
                anns = [bounding_boxes[str(frame_idx)]]
                processed_boxes = self.process_bboxes(w, h, anns)
                bboxes_all_frames.append(processed_boxes)
            else:
                # Append empty list if no bounding box exists for the frame
                bboxes_all_frames.append([])
        # Normalize bounding boxes
        bboxes_all_frames = [np.array(frame_bboxes) for frame_bboxes in bboxes_all_frames]
        bboxes_all_frames = [box_xyxy_to_cxcywh(frame_bboxes) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes_all_frames]
        bboxes_all_frames = [normalize_bboxes(frame_bboxes, w, h) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes_all_frames]
        bboxes_all_frames = [torch.from_numpy(frame_bboxes).float() for frame_bboxes in bboxes_all_frames]

        # Prepare input for Global Image Encoder
        global_enc_images_list_sampled = self.global_enc_processor.preprocess(images_list, return_tensors="pt")[
            "pixel_values"
        ].permute(1, 0, 2, 3).contiguous()

        # Prepare input for Grounding Image Encoder
        images_list = np.stack([self.transform.apply_image(im) for im in images_list])
        grounding_enc_images_list_sampled = self.grounding_enc_processor(
            torch.from_numpy(images_list).permute(3, 0, 1, 2).contiguous()
        )

        # Prepare tagged caption
        questions, conversations = self.create_conversations(caption)

        return (
            video_path,
            global_enc_images_list_sampled,
            grounding_enc_images_list_sampled,
            None,
            conversations,
            bboxes_all_frames,
            (w, h),
            questions,
            [caption],
            temp_objectness_labels
        )

    def __len__(self):
        """
        Total number of samples in the dataset.
        """
        return len(self.annotations["videos"])
    
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.IMG_MEAN[:, None, :, :]) / self.IMG_STD[:, None, :, :]
        _, _, h, w = x.shape
        padding = (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h)
        x = F.pad(x, padding)
        return x