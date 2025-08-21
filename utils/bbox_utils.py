import numpy as np
import torch

def normalize_bboxes(bboxes, image_width, image_height, type='np'):
    """
    Normalize bounding box coordinates.

    Parameters:
    - bboxes (numpy array): Array of bounding boxes in the format [cx, cy, w, h].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - normalized_bboxes (numpy array): Normalized bounding boxes [cx, cy, w, h].
    """
    # Convert the bounding box values from absolute to relative
    normalized_bboxes = np.zeros_like(bboxes) if type == 'np' else torch.zeros_like(bboxes)
    normalized_bboxes[:, 0] = bboxes[:, 0] / image_width   # Normalize cx
    normalized_bboxes[:, 1] = bboxes[:, 1] / image_height  # Normalize cy
    normalized_bboxes[:, 2] = bboxes[:, 2] / image_width   # Normalize width
    normalized_bboxes[:, 3] = bboxes[:, 3] / image_height  # Normalize height

    return normalized_bboxes

def unnormalize_bboxes(normalized_bboxes, image_width, image_height, type='np'):
    """
    Unnormalize bounding box coordinates.

    Parameters:
    - normalized_bboxes (numpy array): Array of normalized bounding boxes in the format [cx, cy, w, h].
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.

    Returns:
    - unnormalized_bboxes (numpy array): Unnormalized bounding boxes [cx, cy, w, h].
    """
    # Convert the bounding box values from relative to absolute
    unnormalized_bboxes = np.zeros_like(normalized_bboxes) if type == 'np' else torch.zeros_like(normalized_bboxes)
    unnormalized_bboxes[:, 0] = normalized_bboxes[:, 0] * image_width   # Unnormalize cx
    unnormalized_bboxes[:, 1] = normalized_bboxes[:, 1] * image_height  # Unnormalize cy
    unnormalized_bboxes[:, 2] = normalized_bboxes[:, 2] * image_width   # Unnormalize width
    unnormalized_bboxes[:, 3] = normalized_bboxes[:, 3] * image_height  # Unnormalize height

    return unnormalized_bboxes

def box_cxcywh_to_xyxy(boxes, type='np'):
    """
    Convert bounding boxes from (cx, cy, w, h) to (x1, y1, x2, y2).

    Parameters:
    boxes (np.ndarray): Array of bounding boxes in format (cx, cy, w, h).

    Returns:
    np.ndarray: Array of bounding boxes in format (x1, y1, x2, y2).
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return np.stack((x1, y1, x2, y2), axis=-1) if type == 'np' else torch.stack((x1, y1, x2, y2), dim=-1)

def box_xyxy_to_cxcywh(boxes, type='np'):
    """
    Convert bounding boxes from (x1, y1, x2, y2) to (cx, cy, w, h).

    Parameters:
    boxes (np.ndarray): Array of bounding boxes in format (x1, y1, x2, y2).

    Returns:
    np.ndarray: Array of bounding boxes in format (cx, cy, w, h).
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    return np.stack((cx, cy, w, h), axis=-1) if type == 'np' else torch.stack((cx, cy, w, h), dim=-1)
