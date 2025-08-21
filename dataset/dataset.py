import numpy as np
import torch

from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from utils.utils import DEFAULT_VIDEO_TOKEN, IGNORE_INDEX, DEFAULT_VID_END_TOKEN, DEFAULT_VID_START_TOKEN


def custom_collate_fn(batch, tokenizer=None, use_mm_start_end=True, inference=False, local_rank=-1):
    # Initializing lists and counters
    video_path_list, global_enc_images_list, grounding_enc_images_list = [], [], []
    bboxes_region_list, conversation_list, bboxes_list = [], [], []
    original_size_list, questions_list = [], []
    selected_labels_list, temp_objectness_labels_list, offset_list, inferences = [], [], [0], []
    cnt = 0

    # Iterating through the batch
    for (video_path, global_enc_images, grounding_enc_images, bboxes_region, conversations, bboxes, original_size, questions,
         data_labels, temp_objectness_labels) in batch:
        video_path_list.append(video_path)
        global_enc_images_list.append(global_enc_images)
        grounding_enc_images_list.append(grounding_enc_images)
        bboxes_region_list.append(bboxes_region)
        conversation_list.extend(conversations)
        bboxes_list.append(bboxes)
        original_size_list.append(original_size)
        questions_list.append(questions)
        selected_labels_list.append(data_labels)
        temp_objectness_labels_list.append(temp_objectness_labels)
        offset_list.append(cnt := cnt + len(conversations))
        inferences.append(inference)

    # Handling the conversation list
    if use_mm_start_end:
        replace_token = DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_TOKEN + DEFAULT_VID_END_TOKEN
        conversation_list = [conv.replace(DEFAULT_VIDEO_TOKEN, replace_token) for conv in conversation_list]

    # Tokenizing and padding input ids
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list],
        batch_first=True, padding_value=tokenizer.pad_token_id
    )
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    # Preparing targets and handling conversation types
    # conv = conversation_lib.default_conversation.copy()
    conv = conversation_lib.conv_templates["llava_v1"].copy()
    targets = input_ids.clone()
    # conv_type == "llava_v1"
    sep = conv.sep + conv.roles[1] + ": "
    sep2 = conv.sep2

    for conversation, target in zip(conversation_list, targets):
        _process_conversation(conversation, target, tokenizer, sep, sep2)

    # Adjusting for inferences
    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 575
        if input_ids.shape[1] > truncate_len:
            input_ids, targets, attention_masks = map(
                lambda x: x[:, :truncate_len], [input_ids, targets, attention_masks]
                )

    return {"video_paths": video_path_list, "global_enc_images": torch.stack(global_enc_images_list, dim=0),
        "grounding_enc_images": torch.stack(grounding_enc_images_list, dim=0),
        "bboxes_region": None if bboxes_region_list[0] is None else bboxes_region_list, "input_ids": input_ids, "labels": targets,
        "attention_masks": attention_masks, "bboxes_list": bboxes_list, "original_size_list": original_size_list,
        "offset": torch.LongTensor(offset_list), "questions_list": questions_list,
        "sampled_classes_list": selected_labels_list, "temp_objectness_labels_list": temp_objectness_labels_list, 
        "inference": inferences[0], "conversation_list": conversation_list, }


def _process_conversation(conversation, target, tokenizer, sep, sep2):
    total_len = target.ne(tokenizer.pad_token_id).sum().item()
    rounds = conversation.split(sep2)
    cur_len = 1
    target[:cur_len] = IGNORE_INDEX

    for rou in rounds:
        if not rou:
            break

        parts = rou.split(sep)
        assert len(parts) == 2, (len(parts), rou)
        parts[0] += sep

        if DEFAULT_VIDEO_TOKEN in conversation:
            round_len = len(tokenizer_image_token(rou, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
        else:
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
        cur_len += round_len

    target[cur_len:] = IGNORE_INDEX
    if cur_len < tokenizer.model_max_length:
        assert cur_len == total_len
