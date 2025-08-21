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
from dataset.utils.utils import VIDEO_GROUNDING_QUESTIONS, VIDEO_STG_QUESTIONS


class HowTo100MDataset(torch.utils.data.Dataset):
    """
    Dataset Class for Video Grounding in HowTo100M dataset.
    """
    CLASSES = ('object',)
    IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    IMG_SIZE = 512
    IGNORE_LABEL = 255

    def __init__(self, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
                 image_size=224, num_classes_per_sample=3, validation=False, random_sampling=False,
                 fps=5, ann_dir='', video_dir='', keys=None):
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

        self.question_templates = VIDEO_GROUNDING_QUESTIONS
        self.begin_str = f"""The {DEFAULT_VIDEO_TOKEN} provides an overview of the video.\n"""
        self.validation = validation

        # Defining paths
        self.ann_dir = ann_dir
        self.video_dir = video_dir
        # self.annotations = self._load_annotations()
        self.keys = keys
        # self.keys = list(self.annotations.keys())

        mode = "Val" if validation else "Train"
        print('\033[92m' + "----GCG-{}: GranDf-GCG dataset initialized----".format(mode) + '\033[0m')

    def _load_annotations(self, video_id):
        with open(os.path.join(self.ann_dir, f"{video_id}.pkl"), "rb") as f:
            annotations = pickle.load(f)
        return annotations

    # def _load_annotations(self):
    #     annotations = pickle.load(open(self.ann_dir, "rb"))
    #     return annotations

    def custom_escape(self, s):
        # Define regex metacharacters that need to be escaped
        metacharacters = ['\\', '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '(', ')', '|']
        escaped = ''
        for char in s:
            if char in metacharacters:
                escaped += '\\' + char
            else:
                escaped += char
        return escaped

    def _parse_annotations(self, ann_info):
        annotations = {'labels': [], 'caption': [], 'bboxes': [], 'temp_objectness_labels': [], 
                       'tokens_positive': [], 'video_fname': ann_info['video_fname'],
                       'fps': ann_info['fps'], 'clip_start_frame': ann_info['clip_start_frame'],
                       'clip_end_frame': ann_info['clip_end_frame'], 'width': ann_info['width'],
                       'height': ann_info['height']}
        orig_caption = ann_info['caption'].strip('"').strip()
        tagged_caption = ann_info['tagged_caption'].strip('"').strip().lower()
        annotations['caption'] = orig_caption.lower()
        # if not self.validation:
            # annotations['labels'] = [p.lower() for p in ann_info['phrases'].tolist()]
        # else:
        annotations['labels'] = [p.lower() for p in ann_info['phrases']]
        # Find positions of phrases in the caption
        for phrase in annotations['labels']:
            # index = annotations['caption'].find(phrase)
            # end_index = index + len(phrase)
            
            pattern = r'<p>' + re.escape(phrase) + r'</p>'
            # pattern = r'<p>' + self.custom_escape(phrase) + r'</p>'
            # Use re.finditer() to find all matches and their positions in the tagged caption
            matches = list(re.finditer(pattern, tagged_caption))
            # If the word is found in the tagged caption
            if matches:
                # Get the start position of the first match
                match = matches[0]
                start_tagged = match.start()
                end_tagged = match.end()

                # Remove the HTML tags to find the corresponding position in the non-tagged caption
                before_tagged = tagged_caption[:start_tagged]
                before_non_tagged = re.sub(r'<.*?>', '', before_tagged)

                # Calculate the start and end positions in the non-tagged caption
                start_non_tagged = len(before_non_tagged)
                end_non_tagged = start_non_tagged + len(phrase)
            
            annotations['tokens_positive'].append([start_non_tagged, end_non_tagged])
        # Sort tokens_positive and video_phrases based on the start index of each phrase
        tokens_positive = annotations['tokens_positive']
        sorted_indices = sorted(range(len(tokens_positive)), key=lambda i: tokens_positive[i][0])
        annotations['tokens_positive'] = [tokens_positive[i] for i in sorted_indices]
        annotations['labels'] = [annotations['labels'][i] for i in sorted_indices]
        num_labels = len(annotations['labels'])

        # if not self.validation:
        #     video_bboxes = [[bbox.tolist() for bbox in frame_bboxes] for frame_bboxes in ann_info['bboxes']]
        #     video_labels = [frame_labels.tolist() for frame_labels in ann_info['labels']]
        # else:
        video_bboxes = [[bbox for bbox in frame_bboxes] for frame_bboxes in ann_info['bboxes']]
        video_labels = [frame_labels for frame_labels in ann_info['labels']]
        for frame_bboxes, frame_labels in zip(video_bboxes, video_labels):
            temp_objectness_labels = np.zeros(num_labels)
            bboxes = []
            label_indices = []
            for bbox, obj_label in zip(frame_bboxes, frame_labels):
                label_index = annotations['labels'].index(obj_label.lower())
                temp_objectness_labels[label_index] = 1
                bboxes.append(bbox)
                label_indices.append(label_index)
            annotations['temp_objectness_labels'].append(temp_objectness_labels)
            # Sort bboxes based on label_indices
            sorted_indices = sorted(range(len(bboxes)), key=lambda i: label_indices[i])
            sorted_bboxes = [bboxes[i] for i in sorted_indices]
            annotations['bboxes'].append(sorted_bboxes)

        # # Trimming overlapping intervals
        # for i in range(len(tokens_positive)):
        #     for j in range(i + 1, len(tokens_positive)):
        #         # If there is overlap
        #         if tokens_positive[i][1] >= tokens_positive[j][0]:
        #             # Modify the end index of phrase i to be one less than the start index of phrase j
        #             tokens_positive[i][1] = tokens_positive[j][0] - 1
        #             # Modify the phrases to reflect the change in indices
        #             annotations['labels'][i] = orig_caption[tokens_positive[i][0]:tokens_positive[i][1] + 1]
        #             break  # Exit inner loop since i was modified

        return annotations
    
    def __getitem__(self, index):
        ann_dict = self._load_annotations(self.keys[index])
        # ann_dict = self.annotations[self.keys[index]]

        # Parse annotation info
        ann = self._parse_annotations(ann_dict)

        return self.process_data(ann)

    def __len__(self):
        return len(self.keys)
    
    def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
        # Normalization: normalize across the spatial dimensions but apply the same mean and std to all frames
        x = (x - self.IMG_MEAN[:, None, :, :]) / self.IMG_STD[:, None, :, :]
        
        # Padding: pad the spatial dimensions to match the desired IMG_SIZE
        # Assuming x has shape [C, T, H, W]
        _, _, h, w = x.shape
        padding = (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h)  # Right and bottom padding
        x = F.pad(x, padding)
        
        return x

    def create_conversations(self, caption, tokens_positive):
        question = random.choice(self.question_templates).strip()

        # Prepare caption with tags
        def tag_caption(caption, tokens):
            for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
                caption = f"{caption[:start]}<p> {caption[start:end]} </p> [DET]{caption[end:]}"
            return caption

        detailed_answer = tag_caption(caption, tokens_positive)

        conversations = []
        # conv = conversation_lib.default_conversation.copy()
        conv = conversation_lib.conv_templates["llava_v1"].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], self.begin_str + question)
        conv.append_message(conv.roles[1], detailed_answer)
        conversations.append(conv.get_prompt())
        questions = [question]
        return questions, conversations
    
    # def sample_frames(self, video_array, num_segments):
    #     """
    #     Sparse sampling of frames from a video array. In 'train' mode, it randomly
    #     samples a frame from each segment. In 'test' mode, it samples the center frame from
    #     each segment.
        
    #     :param video_array: A numpy array of shape (C, T, H, W).
    #     :param num_segments: The number of segments to divide the video into.
    #     :param mode: 'train' for training mode, 'test' for testing mode.
    #     :return: A numpy array of the sampled frames.
    #     """
    #     T = video_array.shape[1]
    #     segment_length = T // num_segments
    #     starts = np.arange(num_segments) * segment_length
    #     ends = np.append(starts[1:], T)

    #     if not self.validation:
    #         # Random offsets within each segment's range
    #         random_offsets = np.random.randint(0, segment_length, size=num_segments)
    #         # Ensure the last segment's offset does not exceed the frame count
    #         random_offsets = np.clip(random_offsets, 0, ends - starts)
    #         indices = starts + random_offsets
    #     else:
    #         # Middle points of each segment
    #         indices = (starts + ends) // 2

    #     return indices, video_array[:, indices, :, :]

    def sample_frames(self, video_array, num_segments):
        """
        Sparse sampling of frames from a video array, with optional importance sampling.
        In 'train' mode, it samples randomly within each segment. In 'test' mode, it
        samples the center frame of each segment.

        :param video_array: A numpy array of shape (C, T, H, W).
        :param num_segments: The number of segments to divide the video into.
        :return: Indices and a numpy array of the sampled frames.
        """
        T = video_array.shape[1]
        segment_length = T // num_segments
        starts = np.arange(num_segments) * segment_length
        ends = np.append(starts[1:], T)

        if not self.validation:
            # Sample randomly within each segment
            indices = []
            for start, end in zip(starts, ends):
                segment_range = np.arange(start, end)
                sampled_index = np.random.randint(start, end)
                indices.append(sampled_index)
            indices = np.array(indices)
        else:
            # Sample midpoint of each segment
            indices = (starts + ends) // 2

        return indices, video_array[:, indices, :, :]

    def process_data(self, data_item):
        data_labels = data_item['labels']
        bboxes = data_item['bboxes']
        temp_objectness_labels = data_item['temp_objectness_labels']
        caption = data_item['caption']
        tokens_positive = data_item['tokens_positive']
        video_path = os.path.join(self.video_dir, data_item['video_fname'])
        video_fps = data_item["fps"]
        start_frame = data_item["clip_start_frame"] 
        end_frame = data_item["clip_end_frame"]
        w = data_item["width"]
        h = data_item["height"]

        # Function to sort elements based on the start index of each phrase
        # def sort_by_start_index(items, order):
        #     return [items[i] for i in order]

        # # Sort phrases based on their appearance in the sentence
        # phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
        # masks = sort_by_start_index(masks, phrase_order)
        # data_labels = sort_by_start_index(data_labels, phrase_order)
        # tokens_positive = sort_by_start_index(tokens_positive, phrase_order)

        sampling_rate = int(video_fps / self.fps)
        frame_ids = list(range(start_frame, end_frame + 1, sampling_rate))
        if len(frame_ids) > len(bboxes):
            frame_ids = list(range(start_frame, end_frame, sampling_rate))
        # ss = start_frame / video_fps
        # t = (end_frame - start_frame) / video_fps
        # cmd = ffmpeg.input(video_path, ss=ss, t=t).filter("fps", fps=len(frame_ids) / t)

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
        # cmd = ffmpeg.input(video_path).filter('select', select=select_expr)
        # out, _ = cmd.output("pipe:", format="rawvideo", pix_fmt="rgb24").run(
        #     capture_stdout=True, quiet=True
        # )
        
        # images_list = np.frombuffer(out, np.uint8).reshape([-1, h, w, 3])
        images_list = np.concatenate([np.frombuffer(frame, np.uint8).reshape([1, h, w, 3]) for frame in all_frames])
        assert len(images_list) == len(frame_ids) == len(bboxes), f"Length mismatch: {len(images_list)}, {len(frame_ids)}, {len(bboxes)}\tVideo: {video_path}"

        # Prepare input for Global Image Encoder
        global_enc_images_list = self.global_enc_processor.preprocess(images_list, return_tensors="pt")["pixel_values"]
        global_enc_images_list = global_enc_images_list.permute(1, 0, 2, 3).contiguous()
        # Prepare input for Grounding Image Encoder
        images_list = np.stack([self.transform.apply_image(im) for im in images_list])
        grounding_enc_images_list = self.grounding_enc_processor(torch.from_numpy(images_list).permute(3, 0, 1, 2).contiguous())
        bboxes_region = None

        questions, conversations = self.create_conversations(caption, tokens_positive)
        # Convert bounding boxes to (c_x, c_y, w, h) format
        bboxes = [np.array(frame_bboxes) for frame_bboxes in bboxes]
        bboxes = [box_xyxy_to_cxcywh(frame_bboxes) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes]
        # Normalize bounding boxes
        bboxes = [normalize_bboxes(frame_bboxes, w, h) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes]
        bboxes = [torch.from_numpy(frame_bboxes).float() for frame_bboxes in bboxes]

        temp_objectness_labels = [torch.from_numpy(frame_temp_objectness_labels) for frame_temp_objectness_labels in temp_objectness_labels]
        original_size = (w, h)

        # Sparse sampling of frames and bounding boxes / labels
        # TODO: add num_segments in the configs
        while True:
            temporal_indices, global_enc_images_list_sampled = self.sample_frames(global_enc_images_list, 8)
            grounding_enc_images_list_sampled = grounding_enc_images_list[:, temporal_indices]
            bboxes_sampled = [bboxes[i] for i in temporal_indices]
            temp_objectness_labels_sampled = [temp_objectness_labels[i] for i in temporal_indices]
            
            if any(temp_objectness.sum() > 0 for temp_objectness in temp_objectness_labels_sampled) or self.validation:
                break

        return (
        video_path, global_enc_images_list_sampled, grounding_enc_images_list_sampled, bboxes_region, conversations, bboxes_sampled, original_size, questions,
        data_labels, temp_objectness_labels_sampled)


# class HowTo100MDataset(torch.utils.data.Dataset):
#     """
#     Dataset Class for Video Grounding in HowTo100M dataset.
#     """
#     CLASSES = ('object',)
#     IMG_MEAN = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
#     IMG_STD = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
#     IMG_SIZE = 512
#     IGNORE_LABEL = 255

#     def __init__(self, tokenizer, global_image_encoder, epoch_samples=8000, precision="fp32",
#                  image_size=224, num_classes_per_sample=3, validation=False, random_sampling=False,
#                  fps=5, ann_dir='', keys=None):
#         self.epoch_samples = epoch_samples
#         self.num_classes_per_sample = num_classes_per_sample
#         self.image_size = image_size
#         self.tokenizer = tokenizer
#         self.precision = precision
#         self.transform = ResizeLongestSide(image_size)
#         self.global_enc_processor = CLIPImageProcessor.from_pretrained(global_image_encoder)
#         self.validation = validation
#         self.random_sampling = random_sampling
#         self.fps = fps

#         self.question_templates = VIDEO_STG_QUESTIONS
#         self.begin_str = f"""The {DEFAULT_VIDEO_TOKEN} provides an overview of the video.\n"""
#         self.validation = validation

#         # Defining paths
#         self.ann_dir = ann_dir
#         # self.annotations = self._load_annotations()
#         self.keys = keys
#         # self.keys = list(self.annotations.keys())

#         mode = "Val" if validation else "Train"
#         print('\033[92m' + "----GCG-{}: GranDf-GCG dataset initialized----".format(mode) + '\033[0m')

#     def _load_annotations(self, video_id):
#         with open(os.path.join(self.ann_dir, f"{video_id}.pkl"), "rb") as f:
#             annotations = pickle.load(f)
#         return annotations

#     def custom_escape(self, s):
#         # Define regex metacharacters that need to be escaped
#         metacharacters = ['\\', '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '(', ')', '|']
#         escaped = ''
#         for char in s:
#             if char in metacharacters:
#                 escaped += '\\' + char
#             else:
#                 escaped += char
#         return escaped

#     def _parse_annotations(self, ann_info):
#         annotations = {'caption': [], 'bboxes': [], 'temp_objectness_labels': [], 
#                        'video_path': ann_info['video_path'],
#                        'fps': ann_info['fps'], 'clip_start_frame': ann_info['clip_start_frame'],
#                        'clip_end_frame': ann_info['clip_end_frame'], 'width': ann_info['width'],
#                        'height': ann_info['height']}
#         orig_caption = ann_info['caption'].strip('"').strip()
#         annotations['caption'] = orig_caption.lower()
#         video_bboxes = [[frame_bboxes] if frame_bboxes is not None else [] for frame_bboxes in ann_info['bboxes']]
#         for frame_bboxes in video_bboxes:
#             temp_objectness_labels = np.zeros(1)
#             if frame_bboxes:
#                 temp_objectness_labels[0] = 1
#             annotations['temp_objectness_labels'].append(temp_objectness_labels)
#         annotations['bboxes'] = video_bboxes

#         return annotations
    
#     def __getitem__(self, index):
#         ann_dict = self._load_annotations(self.keys[index])
#         # ann_dict = self.annotations[self.keys[index]]

#         # Parse annotation info
#         ann = self._parse_annotations(ann_dict)

#         return self.process_data(ann)

#     def __len__(self):
#         return len(self.keys)
    
#     def grounding_enc_processor(self, x: torch.Tensor) -> torch.Tensor:
#         # Normalization: normalize across the spatial dimensions but apply the same mean and std to all frames
#         x = (x - self.IMG_MEAN[:, None, :, :]) / self.IMG_STD[:, None, :, :]
        
#         # Padding: pad the spatial dimensions to match the desired IMG_SIZE
#         # Assuming x has shape [C, T, H, W]
#         _, _, h, w = x.shape
#         padding = (0, self.IMG_SIZE - w, 0, self.IMG_SIZE - h)  # Right and bottom padding
#         x = F.pad(x, padding)
        
#         return x

#     def create_conversations(self, caption):
#         question = random.choice(self.question_templates).strip()

#         detailed_answer = f"<p> {caption} </p> [DET]"

#         conversations = []
#         # conv = conversation_lib.default_conversation.copy()
#         conv = conversation_lib.conv_templates["llava_v1"].copy()
#         conv.messages = []
#         conv.append_message(conv.roles[0], self.begin_str + question)
#         conv.append_message(conv.roles[1], detailed_answer)
#         conversations.append(conv.get_prompt())
#         questions = [question]
#         return questions, conversations

#     def sample_frames(self, video_array, num_segments):
#         """
#         Sparse sampling of frames from a video array, with optional importance sampling.
#         In 'train' mode, it samples randomly within each segment. In 'test' mode, it
#         samples the center frame of each segment.

#         :param video_array: A numpy array of shape (C, T, H, W).
#         :param num_segments: The number of segments to divide the video into.
#         :return: Indices and a numpy array of the sampled frames.
#         """
#         T = video_array.shape[1]
#         segment_length = T // num_segments
#         starts = np.arange(num_segments) * segment_length
#         ends = np.append(starts[1:], T)

#         if not self.validation:
#             # Sample randomly within each segment
#             indices = []
#             for start, end in zip(starts, ends):
#                 segment_range = np.arange(start, end)
#                 sampled_index = np.random.randint(start, end)
#                 indices.append(sampled_index)
#             indices = np.array(indices)
#         else:
#             # Sample midpoint of each segment
#             indices = (starts + ends) // 2

#         return indices, video_array[:, indices, :, :]

#     def process_data(self, data_item):
#         bboxes = data_item['bboxes']
#         temp_objectness_labels = data_item['temp_objectness_labels']
#         caption = data_item['caption']
#         video_path = data_item['video_path'].replace("project_465000880", "project_465001678")
#         video_fps = data_item["fps"]
#         start_frame = data_item["clip_start_frame"] 
#         end_frame = data_item["clip_end_frame"]
#         w = data_item["width"]
#         h = data_item["height"]

#         sampling_rate = int(video_fps / self.fps)
#         frame_ids = list(range(start_frame, end_frame + 1, sampling_rate))
#         if len(frame_ids) > len(bboxes):
#             frame_ids = list(range(start_frame, end_frame, sampling_rate))

#         all_frames = []
#         for frame_id in frame_ids:
#             timestamp = frame_id / video_fps
#             out, _ = (
#                 ffmpeg
#                 .input(video_path, ss=timestamp)
#                 .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
#                 .run(capture_stdout=True, quiet=True)
#             )
#             all_frames.append(out)

#         images_list = np.concatenate([np.frombuffer(frame, np.uint8).reshape([1, h, w, 3]) for frame in all_frames])
#         assert len(images_list) == len(frame_ids) == len(bboxes), f"Length mismatch: {len(images_list)}, {len(frame_ids)}, {len(bboxes)}\tVideo: {video_path}"

#         # Prepare input for Global Image Encoder
#         global_enc_images_list = self.global_enc_processor.preprocess(images_list, return_tensors="pt")["pixel_values"]
#         global_enc_images_list = global_enc_images_list.permute(1, 0, 2, 3).contiguous()
#         # Prepare input for Grounding Image Encoder
#         images_list = np.stack([self.transform.apply_image(im) for im in images_list])
#         grounding_enc_images_list = self.grounding_enc_processor(torch.from_numpy(images_list).permute(3, 0, 1, 2).contiguous())
#         bboxes_region = None

#         questions, conversations = self.create_conversations(caption)
#         # Convert bounding boxes to (c_x, c_y, w, h) format
#         bboxes = [np.array(frame_bboxes) for frame_bboxes in bboxes]
#         bboxes = [box_xyxy_to_cxcywh(frame_bboxes) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes]
#         # Normalize bounding boxes
#         bboxes = [normalize_bboxes(frame_bboxes, w, h) if frame_bboxes.size else frame_bboxes for frame_bboxes in bboxes]
#         bboxes = [torch.from_numpy(frame_bboxes).float() for frame_bboxes in bboxes]

#         temp_objectness_labels = [torch.from_numpy(frame_temp_objectness_labels) for frame_temp_objectness_labels in temp_objectness_labels]
#         original_size = (w, h)

#         # Sparse sampling of frames and bounding boxes / labels
#         # TODO: add num_segments in the configs
#         while True:
#             temporal_indices, global_enc_images_list_sampled = self.sample_frames(global_enc_images_list, 8)
#             grounding_enc_images_list_sampled = grounding_enc_images_list[:, temporal_indices]
#             bboxes_sampled = [bboxes[i] for i in temporal_indices]
#             temp_objectness_labels_sampled = [temp_objectness_labels[i] for i in temporal_indices]
            
#             if any(temp_objectness.sum() > 0 for temp_objectness in temp_objectness_labels_sampled) or self.validation:
#                 break

#         return (
#         video_path, global_enc_images_list_sampled, grounding_enc_images_list_sampled, bboxes_region, conversations, bboxes_sampled, original_size, questions,
#         None, temp_objectness_labels_sampled)
