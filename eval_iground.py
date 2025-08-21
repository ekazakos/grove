import os
import json
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocoevalcap.eval import COCOEvalCap

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation script with options for flattening and per-video evaluation")
    parser.add_argument("--split", required=True, help="Evaluation split, options are 'val', 'test'")
    parser.add_argument("--pred_file_path", required=True, help="The path where the inference results are stored.")
    parser.add_argument("--gt_file_path", required=False, default="./data/GranDf/annotations/val_test",
                        help="The path containing GranD-f evaluation annotations.")
    parser.add_argument("--save_dir", required=False, default="./coco_format_data", help="The directory to save the COCO format data.")
    parser.add_argument("--evaluation_mode", required=False, choices=["flattening", "per_video"], default="flattening",
                        help="Evaluation mode, options are 'flattening' or 'per_video'.")
    args = parser.parse_args()
    return args

# Load pre-trained model tokenizer and model for evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()
model.eval()

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    # Use the mean of the last hidden states as sentence embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().cpu().numpy()
    return sentence_embedding

def compute_iou(pred_bbox, gt_bbox):
    # Compute IoU between two bounding boxes
    xA = max(pred_bbox[0], gt_bbox[0])
    yA = max(pred_bbox[1], gt_bbox[1])
    xB = min(pred_bbox[2], gt_bbox[2])
    yB = min(pred_bbox[3], gt_bbox[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (pred_bbox[2] - pred_bbox[0] + 1) * (pred_bbox[3] - pred_bbox[1] + 1)
    boxBArea = (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1)

    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def compute_iou_matrix(pred_bboxes, gt_bboxes):
    iou_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i, pred_bbox in enumerate(pred_bboxes):
        for j, gt_bbox in enumerate(gt_bboxes):
            iou_matrix[i, j] = compute_iou(pred_bbox, gt_bbox)
    return iou_matrix

def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)
    return cosine_similarity([emb1], [emb2])[0, 0]

def find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold):
    best_matches = []

    # Compute pair-wise IoU
    pred_bboxes = [ann['bbox'] for ann in dt_anns]
    gt_bboxes = [ann['bbox'] for ann in gt_anns]
    ious = compute_iou_matrix(gt_bboxes, pred_bboxes)

    text_sims = np.zeros((len(gt_labels), len(dt_labels)))

    for i, gt_label in enumerate(gt_labels):
        for j, dt_label in enumerate(dt_labels):
            text_sims[i, j] = text_similarity_bert(gt_label, dt_label)

    # Find one-to-one matches satisfying both IoU and text similarity thresholds
    while ious.size > 0 and text_sims.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break  # No admissible pair found

        best_matches.append(max_iou_idx)

        # Remove selected annotations from consideration
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]

def evaluate_bbox_miou(coco_gt, video_ids, pred_save_path, mode):
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_save_path)

    if mode == "flattening":
        mious = []
        for video_id in tqdm(video_ids):
            frame_ids = [frame['id'] for frame in coco_gt.imgs.values() if frame['video_id'] == video_id]

            for frame_id in frame_ids:
                matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == frame_id]
                ann_ids = [ann['id'] for ann in matching_anns]

                gt_anns = coco_gt.loadAnns(ann_ids)
                gt_bboxes = [ann['bbox'] for ann in gt_anns if 'bbox' in ann]

                matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == frame_id]
                dt_ann_ids = [ann['id'] for ann in matching_anns]
                pred_anns = coco_dt.loadAnns(dt_ann_ids)
                pred_bboxes = [ann['bbox'] for ann in pred_anns if 'bbox' in ann]

                if pred_bboxes and gt_bboxes:
                    mious.append(compute_iou_matrix(pred_bboxes, gt_bboxes).mean())

        mean_miou = np.mean(mious) if mious else 0.0

    else:
        video_mious = []
        for video_id in tqdm(video_ids):
            frame_mious = []
            frame_ids = [frame['id'] for frame in coco_gt.imgs.values() if frame['video_id'] == video_id]

            for frame_id in frame_ids:
                matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == frame_id]
                ann_ids = [ann['id'] for ann in matching_anns]

                gt_anns = coco_gt.loadAnns(ann_ids)
                gt_bboxes = [ann['bbox'] for ann in gt_anns if 'bbox' in ann]

                matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == frame_id]
                dt_ann_ids = [ann['id'] for ann in matching_anns]
                pred_anns = coco_dt.loadAnns(dt_ann_ids)
                pred_bboxes = [ann['bbox'] for ann in pred_anns if 'bbox' in ann]

                if pred_bboxes and gt_bboxes:
                    frame_mious.append(compute_iou_matrix(pred_bboxes, gt_bboxes).mean())

            video_miou = np.mean(frame_mious) if frame_mious else 0.0
            video_mious.append(video_miou)

        mean_miou = np.mean(video_mious) if video_mious else 0.0

    print(f"Mean IoU (mIoU) across all videos: {mean_miou:.3f}")

def evaluate_recall_with_mapping(coco_gt, coco_cap_gt, video_ids, pred_save_path, cap_pred_save_path, iou_threshold=0.5,
                                 text_sim_threshold=0.5, mode="flattening"):
    coco_dt = coco_gt.loadRes(pred_save_path)
    coco_cap_dt = coco_cap_gt.loadRes(cap_pred_save_path)

    if mode == "flattening":
        true_positives = 0
        actual_positives = 0

        for video_id in tqdm(video_ids):
            frame_ids = [frame['id'] for frame in coco_gt.imgs.values() if frame['video_id'] == video_id]

            for frame_id in frame_ids:
                try:
                    matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == frame_id]
                    gt_ann_ids = [ann['id'] for ann in matching_anns]
                    gt_anns = coco_gt.loadAnns(gt_ann_ids)

                    matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == frame_id]
                    dt_ann_ids = [ann['id'] for ann in matching_anns]
                    dt_anns = coco_dt.loadAnns(dt_ann_ids)

                    matching_anns = [ann for ann in coco_cap_gt.anns.values() if ann['image_id'] == frame_id]
                    gt_cap_ann_ids = [ann['id'] for ann in matching_anns]
                    gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ann_ids)[0]
                    
                    matching_anns = [ann for ann in coco_cap_dt.anns.values() if ann['image_id'] == frame_id]
                    dt_cap_ann_ids = [ann['id'] for ann in matching_anns]
                    dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ann_ids)[0]

                    gt_labels = gt_cap_ann['labels']
                    dt_labels = dt_cap_ann['labels']

                    actual_positives += len(gt_labels)

                    best_matches = find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold)

                    true_positives += len(best_matches)
                except Exception as e:
                    print(e)

        recall = true_positives / actual_positives if actual_positives > 0 else 0
        print(f"Recall: {recall:.3f}")

    else:
        video_recalls = []
        for video_id in tqdm(video_ids):
            video_true_positives = 0
            video_actual_positives = 0
            frame_ids = [frame['id'] for frame in coco_gt.imgs.values() if frame['video_id'] == video_id]

            for frame_id in frame_ids:
                try:
                    matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == frame_id]
                    gt_ann_ids = [ann['id'] for ann in matching_anns]
                    gt_anns = coco_gt.loadAnns(gt_ann_ids)

                    matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == frame_id]
                    dt_ann_ids = [ann['id'] for ann in matching_anns]
                    dt_anns = coco_dt.loadAnns(dt_ann_ids)

                    matching_anns = [ann for ann in coco_cap_gt.anns.values() if ann['image_id'] == frame_id]
                    gt_cap_ann_ids = [ann['id'] for ann in matching_anns]
                    gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ann_ids)[0]

                    matching_anns = [ann for ann in coco_cap_dt.anns.values() if ann['image_id'] == frame_id]
                    dt_cap_ann_ids = [ann['id'] for ann in matching_anns]
                    dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ann_ids)[0]

                    gt_labels = gt_cap_ann['labels']
                    dt_labels = dt_cap_ann['labels']

                    video_actual_positives += len(gt_labels)

                    best_matches = find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold)

                    video_true_positives += len(best_matches)
                except Exception as e:
                    print(e)

            video_recall = video_true_positives / video_actual_positives if video_actual_positives > 0 else 0
            video_recalls.append(video_recall)

        mean_recall = np.mean(video_recalls) if video_recalls else 0.0
        print(f"Recall: {mean_recall:.3f}")

def evaluate_ap_per_video(coco_gt, coco_dt, video_ids, cat_ids=[1], mode="flattening"):
    if mode == "flattening":
        # Evaluate for the aggregated frames
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mean_ap = coco_eval.stats[0]  # AP @ IoU=0.50:0.95 area=all maxDets=100
    else:
        video_ap_scores = []
        for video_id in tqdm(video_ids):
            # Aggregate results for the current video
            frame_ids = [frame['id'] for frame in coco_gt.imgs.values() if frame['video_id'] == video_id]
            # Evaluate for the aggregated frames
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.catIds = cat_ids
            coco_eval.params.imgIds = frame_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            video_ap_scores.append(coco_eval.stats[1])  # AP @ IoU=0.50:0.95 area=all maxDets=100

        mean_ap = np.mean(video_ap_scores) if video_ap_scores else 0.0
        print(f"Mean AP (mAP) across all videos: {mean_ap:.3f}")

    return mean_ap

def sample_frames(data, num_segments):
        """
        Sparse sampling of frames from a video array. In 'train' mode, it randomly
        samples a frame from each segment. In 'test' mode, it samples the center frame from
        each segment.
        
        :param video_array: A numpy array of shape (C, T, H, W).
        :param num_segments: The number of segments to divide the video into.
        :param mode: 'train' for training mode, 'test' for testing mode.
        :return: A numpy array of the sampled frames.
        """
        T = len(data)
        segment_length = T // num_segments
        starts = np.arange(num_segments) * segment_length
        ends = np.append(starts[1:], T)

        # Middle points of each segment
        indices = (starts + ends) // 2

        return indices, [data[i] for i in indices]

def clamp_bbox(bbox, width, height):
    # Separate x and y coordinates
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)  # Clamp x_min and x_max
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height) # Clamp y_min and y_max
    return bbox

def transform_data_to_coco_format(gt_data, pred_data, save_dir):
    coco_gt_bbox = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "object"}],
    }
    coco_gt_caption = {
        "images": [],
        "annotations": [],
    }
    coco_pred_bbox = []
    coco_pred_caption = []
    
    coco_gt_labels = {
        "images": [],
        "annotations": [],
    }
    coco_pred_labels = []

    ann_id = 0
    caption_ann_id = 0  # Separate counter for caption annotations
    label_ann_id = 0    # Separate counter for label annotations

    for video_id, gt_video_data in gt_data.items():
        if video_id not in pred_data:
            print(f"Warning: {video_id} not found in predictions.")
            continue
        pred_video_data = pred_data.get(video_id, {})
        
        gt_bboxes = gt_video_data.get('bboxes', [])
        # indices, gt_bboxes = sample_frames(gt_bboxes, 8)
        gt_labels = gt_video_data.get('labels', [])
        # gt_labels = [gt_labels[i] for i in indices]
        gt_text = gt_video_data.get('caption', "")
        gt_phrases = gt_video_data.get('phrases', [])
        
        pred_bboxes = pred_video_data.get('pred_bboxes', [])
        if len(pred_bboxes) > len(gt_bboxes):
            pred_bboxes = pred_bboxes[:-1]
        # pred_bboxes = [pred_bboxes[i] for i in indices]
        pred_labels = pred_video_data.get('pred_labels', [])
        if len(pred_labels) > len(gt_labels):
            pred_labels = pred_labels[:-1]
        # pred_labels = [pred_labels[i] for i in indices]
        pred_text = pred_video_data.get('pred_text', "")
        pred_phrases = pred_video_data.get('pred_phrases', [])
        if len(pred_bboxes) == 0 and len(pred_labels) == 0:
            continue
        assert len(gt_bboxes) == len(gt_labels) == len(pred_bboxes) == len(pred_labels), f"Mismatch in number of bounding boxes and labels between GT and predictions, video_id: {video_id}, gt_bboxes: {len(gt_bboxes)}, gt_labels: {len(gt_labels)}, pred_bboxes: {len(pred_bboxes)}, pred_labels: {len(pred_labels)}"

        for frame_idx, (frame_gt_bboxes, frame_gt_labels, frame_pred_bboxes, frame_pred_labels) in enumerate(zip(gt_bboxes, gt_labels, pred_bboxes, pred_labels)):
            image_id = f"{video_id}_{frame_idx}"
            coco_gt_bbox["images"].append({"id": str(image_id), "video_id": str(video_id), "frame_idx": int(frame_idx)})
            coco_gt_labels["images"].append({"id": str(image_id), "video_id": str(video_id), "frame_idx": int(frame_idx)})

            if frame_idx == 0:
                coco_gt_caption["images"].append({"id": str(image_id), "video_id": str(video_id), "frame_idx": int(frame_idx)})

            for bbox, label in zip(frame_gt_bboxes, frame_gt_labels):
                x_min, y_min, x_max, y_max = map(int, bbox)
                width = x_max - x_min
                height = y_max - y_min
                area = width * height
                ann_id += 1
                coco_gt_bbox["annotations"].append({
                    "id": int(ann_id),
                    "image_id": str(image_id),
                    "category_id": 1,
                    "bbox": [x_min, y_min, width, height],
                    "area": area,
                    "iscrowd": 0,
                })

            for bbox, label in zip(frame_pred_bboxes, frame_pred_labels):
                bbox = clamp_bbox(bbox, gt_video_data['width'], gt_video_data['height'])
                x_min, y_min, x_max, y_max = map(int, bbox)
                width = x_max - x_min
                height = y_max - y_min
                coco_pred_bbox.append({
                    "image_id": str(image_id),
                    "category_id": 1,
                    "bbox": [x_min, y_min, width, height],
                    "score": 1.0,
                })

            label_ann_id += 1
            coco_gt_labels["annotations"].append({
                "image_id": str(image_id),
                "id": int(label_ann_id),
                "caption": gt_text,
                "labels": frame_gt_labels,
            })

            coco_pred_labels.append({
                "image_id": str(image_id),
                "caption": pred_text,
                "labels": frame_pred_labels,
            })

        caption_ann_id += 1
        coco_gt_caption["annotations"].append({
            "image_id": str(f"{video_id}_0"),  # Use the first frame's id for the caption
            "id": int(caption_ann_id),
            "caption": gt_text,
            "labels": gt_phrases,
        })

        coco_pred_caption.append({
            "image_id": str(f"{video_id}_0"),  # Use the first frame's id for the caption
            "caption": pred_text,
            "labels": pred_phrases,
        })

    # Save the JSON files
    with open(f"{save_dir}/gt_bbox.json", 'w') as f:
        json.dump(coco_gt_bbox, f)
    with open(f"{save_dir}/gt_caption.json", 'w') as f:
        json.dump(coco_gt_caption, f)
    with open(f"{save_dir}/pred_bbox.json", 'w') as f:
        json.dump(coco_pred_bbox, f)
    with open(f"{save_dir}/pred_caption.json", 'w') as f:
        json.dump(coco_pred_caption, f)
    with open(f"{save_dir}/gt_labels.json", 'w') as f:
        json.dump(coco_gt_labels, f)
    with open(f"{save_dir}/pred_labels.json", 'w') as f:
        json.dump(coco_pred_labels, f)

def main():
    args = parse_args()

    print(f"Starting evaluation on {args.split} split.")

    # Transform your data into COCO format
    with open(args.gt_file_path, 'rb') as f:
        gt_data = pickle.load(f)
        print(len(gt_data.keys()))

    with open(args.pred_file_path, 'rb') as f:
        pred_data = pickle.load(f)
        print(len(pred_data.keys()))

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    transform_data_to_coco_format(gt_data, pred_data, save_dir)

    gt_bbox_path = f"{save_dir}/gt_bbox.json"
    gt_cap_path = f"{save_dir}/gt_caption.json"
    pred_bbox_path = f"{save_dir}/pred_bbox.json"
    pred_cap_path = f"{save_dir}/pred_caption.json"
    gt_labels_path = f"{save_dir}/gt_labels.json"
    pred_labels_path = f"{save_dir}/pred_labels.json"

    # Get the video IDs for the split
    all_video_ids = []
    with open(gt_cap_path, 'r') as f:
        contents = json.load(f)
        for video in contents['images']:
            all_video_ids.append(video['video_id'])

    # # -------------------------------#
    # 1. Evaluate AP
    coco_gt = COCO(gt_bbox_path)
    coco_dt = coco_gt.loadRes(pred_bbox_path)
    evaluate_ap_per_video(coco_gt, coco_dt, all_video_ids, mode=args.evaluation_mode)

    # # -------------------------------#
    # # 2. Evaluate Caption Quality
    coco_cap_gt = COCO(gt_cap_path)
    coco_cap_result = coco_cap_gt.loadRes(pred_cap_path)
    coco_eval = COCOEvalCap(coco_cap_gt, coco_cap_result)
    coco_eval.params['image_id'] = coco_cap_result.getImgIds()
    coco_eval.evaluate()
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')

    # # -------------------------------#
    # 3. Evaluate Bounding Box Mean IoU
    evaluate_bbox_miou(coco_gt, all_video_ids, pred_bbox_path, mode=args.evaluation_mode)

    # # -------------------------------#
    # 4. Evaluate Recall
    coco_cap_gt_labels = COCO(gt_labels_path)
    coco_cap_pred_labels = coco_cap_gt_labels.loadRes(pred_labels_path)
    evaluate_recall_with_mapping(coco_gt, coco_cap_gt_labels, all_video_ids, pred_bbox_path, pred_labels_path, iou_threshold=0.5, text_sim_threshold=0.5, mode=args.evaluation_mode)


if __name__ == "__main__":
    main()