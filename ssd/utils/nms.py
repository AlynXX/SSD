import torch
from torchvision.ops import nms as torchvision_nms

def nms(boxes, scores, iou_threshold):
    """
    Wykonuje Non-Maximum Suppression (NMS) na bounding boxach.
    
    Args:
        boxes (Tensor): Tensor o kształcie [N, 4] z współrzędnymi [x1, y1, x2, y2].
        scores (Tensor): Tensor o kształcie [N] z wartościami ufności.
        iou_threshold (float): Próg IoU dla NMS.
    
    Returns:
        keep (Tensor): Indeksy bounding boxów, które pozostają po NMS.
    """
    return torchvision_nms(boxes, scores, iou_threshold)

def batched_nms(boxes, scores, idxs, iou_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Parameters
    ----------
    boxes : Tensor[N, 4]
        boxes where NMS will be performed. They
        are expected to be in (x1, y1, x2, y2) format
    scores : Tensor[N]
        scores for each one of the boxes
    idxs : Tensor[N]
        indices of the categories for each one of the boxes.
    iou_threshold : float
        discards all overlapping boxes
        with IoU < iou_threshold

    Returns
    -------
    keep : Tensor
        int64 tensor with the indices of
        the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=boxes.device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = torchvision_nms(boxes_for_nms, scores, iou_threshold)
    return keep