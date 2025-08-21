import argparse
import pickle
import os
import numpy as np

import numpy as np

def evaluate_dataset_localization(pred_boxes_dict, gt_data, dataset):
    """
    Evaluate localization accuracy across multiple videos.

    Args:
        pred_boxes_dict (dict): {video_id: list of predicted (xtl, ytl, xbr, ybr) or None}
        gt_boxes_dict (dict): {video_id: {frame_id: (xtl, ytl, xbr, ybr, outside, occluded)}}

    Returns:
        float: overall accuracy percentage
        int: total correct predictions
        int: total valid ground truth boxes
    """
    total_correct = 0
    total_valid = 0

    for gt_clip_data in gt_data:
        video_id = gt_clip_data["video_id"]
        segment_id = gt_clip_data[f"segment_{dataset}_idx"]
        unique_id = f"{video_id}_{segment_id}"
        pred_boxes = pred_boxes_dict.get(unique_id, []).get("final_boxes", [])
        gt_boxes = gt_clip_data["segment_bboxes"]

        for pred_box, gt_box in zip(pred_boxes, gt_boxes):
            # Check validity
            if not gt_box:
                continue

            xtl, ytl, xbr, ybr = gt_box

            total_valid += 1

            if pred_box is None or np.any(np.isnan(pred_box)):
                continue

            pred_box = pred_box[0]
            cx = (pred_box[0] + pred_box[2]) / 2
            cy = (pred_box[1] + pred_box[3]) / 2

            if xtl <= cx <= xbr and ytl <= cy <= ybr:
                total_correct += 1

    accuracy = (total_correct / total_valid) * 100 if total_valid else 0.0
    return accuracy, total_correct, total_valid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True, help="Path to the predictions file.")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to the ground truth file.")
    parser.add_argument("--dataset", type=str, default="youcookinteractions", choices=['youcook', 'groundingyoutube'], help="Dataset name.")

    args = parser.parse_args()
    predictions_path = args.predictions
    ground_truth_path = args.ground_truth
    dataset = args.dataset

    # Load predictions
    with open(predictions_path, "rb") as f:
        pred_boxes_dict = pickle.load(f)

    # Load ground truth data
    with open(ground_truth_path, "rb") as f:
        gt_data = pickle.load(f)

    # Evaluate localization accuracy
    accuracy, total_correct, total_valid = evaluate_dataset_localization(pred_boxes_dict, gt_data, dataset)
    print(f"Localization Accuracy: {accuracy:.2f}%")