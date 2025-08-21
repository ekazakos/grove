import pickle
from pathlib import Path
from typing import Dict, List
from argparse import ArgumentParser

import numpy as np

import json
from functools import reduce
from typing import Tuple

#### Bounding box utilities imported from torchvision and converted to numpy
def np_box_area(boxes: np.array) -> np.array:
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format with
            ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Returns:
        area (Tensor[N]): area for each box
    """
    assert boxes.ndim == 2 and boxes.shape[-1] == 4, f"Boxes should have 2 dimensions and the last dimension should be 4, got {boxes.shape}"
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def _box_inter_union(boxes1: np.array, boxes2: np.array) -> Tuple[np.array, np.array]:
    area1 = np_box_area(boxes1)
    area2 = np_box_area(boxes2)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


def np_box_iou(boxes1: np.array, boxes2: np.array) -> np.array:
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        boxes1 (Tensor[N, 4])
        boxes2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values for every element in boxes1 and boxes2
    """
    inter, union = _box_inter_union(boxes1, boxes2)
    iou = inter / union
    return iou


class VidSTGiouEvaluator:
    def __init__(
        self,
        vidstg_path: str,
        subset: str = "test",
        verbose: bool = True,
        iou_thresholds: list = [0.3, 0.5],
        fps: int = 10,
    ):
        """
        :param vidstg_path: path to VidSTG annotations
        :param subset: train, val or test
        :param verbose: whether to print more information or not
        :param iou_thresholds: IoU thresholds for the vIoU metrics
        :param fps: number of frames per second
        """
        assert subset in ["train", "test", "val"], f"Wrong VidSTG subset {subset}"

        self.iou_thresholds = iou_thresholds

        vidstg_path = Path(vidstg_path)

        # Load and prepare ground truth data
        path = vidstg_path / f"{subset}.json"
        self.anns = json.load(open(path, "r"))
        self.video_gt = {}  # Ground truth for each video

        for video in self.anns["videos"]:
            video_id = video["video_id"]
            video_fps = video["fps"]
            sampling_rate = int(video_fps / fps)
            start_frame = video["tube_start_frame"]
            end_frame = video["tube_end_frame"]

            frame_ids = list(range(start_frame, end_frame - 1, sampling_rate))
            boxes = []
            for frame_id in frame_ids:
                if video["tube_start_frame"] <= frame_id < video["tube_end_frame"]:
                    x1, y1, w, h = self.anns["trajectories"][
                        video["original_video_id"]
                    ][str(video["target_id"])][str(frame_id)]["bbox"]
                    x2 = x1 + w
                    y2 = y1 + h
                    boxes.append([x1, y1, x2, y2])
                else:
                    boxes.append([])

            self.video_gt[video_id] = {"frame_ids": frame_ids, "boxes": boxes}

        if verbose:
            print(f"VidSTG subset contains {len(self.video_gt)} videos with GT annotations.")

    def evaluate(self, video_predictions: Dict[str, Dict]):
        """
        Evaluate video-level predictions against ground truth.

        :param video_predictions: Dictionary of video predictions in the specified structure.
        :return: Metrics for each video.
        """
        vid_metrics = {}

        # Check for missing video predictions
        missing_videos = [video_id for video_id in self.video_gt if video_id not in video_predictions]
        if missing_videos:
            raise RuntimeError(f"Missing predictions for {len(missing_videos)} videos: {missing_videos}")

        for video_id, gt_data in self.video_gt.items():
            if video_id not in video_predictions:
                raise RuntimeError(f"Missing prediction for video {video_id}")

            pred_data = video_predictions[video_id]
            pred_boxes = pred_data["boxes"]
            pred_frame_ids = pred_data["frame_ids"]

            gt_boxes = gt_data["boxes"]
            gt_frame_ids = gt_data["frame_ids"]

            if len(pred_boxes) != len(pred_frame_ids):
                raise ValueError(f"Inconsistent predictions for video {video_id}: number of boxes and frame IDs do not match.")

            vid_metrics[video_id] = {
                "qtype": pred_data["qtype"],
                "img_metrics": {},
            }

            gt_viou = 0

            # Iterate over GT frames
            for gt_frame_idx, gt_box in zip(gt_frame_ids, gt_boxes):
                if gt_frame_idx not in pred_frame_ids:
                    raise RuntimeError(f"Missing prediction for frame {gt_frame_idx} in video {video_id}")

                # Find the corresponding predicted box
                pred_idx = pred_frame_ids.index(gt_frame_idx)
                pred_box = pred_boxes[pred_idx]
                if pred_box.any():
                    iou = np_box_iou(np.array(pred_box), np.array([gt_box]))[0][0]
                else:
                    iou = 0
                vid_metrics[video_id]["img_metrics"][f"{video_id}_{gt_frame_idx}"] = {
                    "iou": iou,
                    "pred_box": pred_box,
                    "gt_box": gt_box,
                }
                gt_viou += iou

            # Compute gt_viou@R
            gt_viou = gt_viou / max(len(gt_frame_ids), 1)
            vid_metrics[video_id]["gt_viou"] = gt_viou
            gt_recalls = {thresh: 0 for thresh in self.iou_thresholds}
            for thresh in self.iou_thresholds:
                if gt_viou > thresh:
                    gt_recalls[thresh] += 1
            vid_metrics[video_id].update(
                {
                    f"gt_viou@{thresh}": gt_recalls[thresh]
                    for thresh in self.iou_thresholds
                }
            )

        return vid_metrics


class VidSTGEvaluator:
    def __init__(
        self,
        vidstg_path,
        subset,
        video_predictions,
        iou_thresholds=[0.3, 0.5],
        fps=5,
    ):
        """
        :param vidstg_path: path to VidSTG annotations
        :param subset: train, val or test
        :param video_predictions: Predictions for videos in the specified structure.
        :param iou_thresholds: IoU thresholds for the vIoU metrics.
        :param fps: Frames per second for evaluation.
        """
        self.evaluator = VidSTGiouEvaluator(
            vidstg_path,
            subset=subset,
            verbose=False,
            iou_thresholds=iou_thresholds,
            fps=fps,
        )
        self.video_predictions = pickle.load(open(video_predictions, "rb"))
        self.results = None
        self.iou_thresholds = iou_thresholds

    def summarize(self):
        """
        Summarize evaluation metrics.
        """
        self.results = self.evaluator.evaluate(self.video_predictions)
        categories = set(x["qtype"] for x in self.results.values())
        metrics = {}
        counter = {}
        for category in categories:  # Initialize metrics
            metrics[category] = {"gt_viou": 0}
            for thresh in self.iou_thresholds:
                metrics[category][f"gt_viou@{thresh}"] = 0
            counter[category] = 0
        for x in self.results.values():  # Aggregate results
            qtype = x["qtype"]
            metrics[qtype]["gt_viou"] += x["gt_viou"]
            for thresh in self.iou_thresholds:
                metrics[qtype][f"gt_viou@{thresh}"] += x[f"gt_viou@{thresh}"]
            counter[qtype] += 1
        for category in categories:  # Average results
            for key in metrics[qtype]:
                metrics[category][key] = metrics[category][key] / counter[category]
                print(f"{category} {key}: {metrics[category][key]:.4f}")
        out = {
            f"{qtype}_{name}": metrics[qtype][name]
            for qtype in metrics
            for name in metrics[qtype]
        }

        return out
    

def main(args):
    vidstg_evaluator = VidSTGEvaluator(
        vidstg_path=args.vidstg_path,
        subset=args.subset,
        video_predictions=args.video_predictions,
    )

    results = vidstg_evaluator.summarize()
    print(results)

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate VidSTG predictions for spatial grounding.")
    parser.add_argument("--vidstg_path", type=str, default="/home/VidSTG/annotations/", help="Path to VidSTG annotations")
    parser.add_argument("--subset", type=str, default="test", help="Subset to evaluate: train, val or test")
    parser.add_argument("--video_predictions", type=str, default="/home/grove_inference_output/result_vidstg.pkl", help="Path to video predictions")
    args = parser.parse_args()

    main(args)