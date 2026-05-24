# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from PaddlePaddle/PaddleX (Apache-2.0):
# this file is a cherry-pick of the eight helper functions from
# paddlex/inference/pipelines/layout_parsing/utils.py that are transitively
# needed by xycut_enhanced. Function bodies are verbatim; only the surrounding
# (unrelated) helpers were dropped to avoid pulling in OCR/component imports.

import re
from typing import List, Union

import numpy as np


def calculate_projection_overlap_ratio(
    bbox1: List[float],
    bbox2: List[float],
    direction: str = "horizontal",
    mode="union",
) -> float:
    """
    Calculate the IoU of lines between two bounding boxes.

    Args:
        bbox1 (List[float]): First bounding box [x_min, y_min, x_max, y_max].
        bbox2 (List[float]): Second bounding box [x_min, y_min, x_max, y_max].
        direction (str): direction of the projection, "horizontal" or "vertical".

    Returns:
        float: Line overlap ratio. Returns 0 if there is no overlap.
    """
    start_index, end_index = 1, 3
    if direction == "horizontal":
        start_index, end_index = 0, 2

    intersection_start = max(bbox1[start_index], bbox2[start_index])
    intersection_end = min(bbox1[end_index], bbox2[end_index])
    overlap = intersection_end - intersection_start
    if overlap <= 0:
        return 0

    if mode == "union":
        ref_width = max(bbox1[end_index], bbox2[end_index]) - min(
            bbox1[start_index], bbox2[start_index]
        )
    elif mode == "small":
        ref_width = min(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    elif mode == "large":
        ref_width = max(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    return overlap / ref_width if ref_width > 0 else 0.0


def calculate_overlap_ratio(
    bbox1: Union[np.ndarray, list, tuple],
    bbox2: Union[np.ndarray, list, tuple],
    mode="union",
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes using NumPy.

    Args:
        bbox1 (np.ndarray, list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (np.ndarray, list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    bbox1 = np.array(bbox1, dtype=np.float64)
    bbox2 = np.array(bbox2, dtype=np.float64)

    x_min_inter = np.maximum(bbox1[0], bbox2[0])
    y_min_inter = np.maximum(bbox1[1], bbox2[1])
    x_max_inter = np.minimum(bbox1[2], bbox2[2])
    y_max_inter = np.minimum(bbox1[3], bbox2[3])

    inter_width = np.maximum(0, x_max_inter - x_min_inter)
    inter_height = np.maximum(0, y_max_inter - y_min_inter)

    inter_area = np.multiply(inter_width, inter_height, dtype=np.float64)

    bbox1_area = calculate_bbox_area(bbox1)
    bbox2_area = calculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = np.minimum(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = np.maximum(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area


def is_english_letter(char):
    """check if the char is english letter"""
    return bool(re.match(r"^[A-Za-z]$", char))


def is_numeric(char):
    """check if the char is numeric"""
    return bool(re.match(r"^[\d]+$", char))


def is_non_breaking_punctuation(char):
    """
    check if the char is non-breaking punctuation

    Args:
        char (str): character to check

    Returns:
        bool: True if the char is non-breaking punctuation
    """
    non_breaking_punctuations = {
        ",",
        "，",
        "、",
        ";",
        "；",
        ":",
        "：",
        "-",
        "'",
        '"',
        "“",
    }

    return char in non_breaking_punctuations


def calculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area


def caculate_euclidean_dist(point1, point2):
    """Calculate euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_seg_flag(block, prev_block):
    """Get segment start flag and end flag based on previous block

    Args:
        block (Block): Current block
        prev_block (Block): Previous block

    Returns:
        seg_start_flag (bool): Segment start flag
        seg_end_flag (bool): Segment end flag
    """

    seg_start_flag = True
    seg_end_flag = True

    context_left_coordinate = block.start_coordinate
    context_right_coordinate = block.end_coordinate
    seg_start_coordinate = block.seg_start_coordinate
    seg_end_coordinate = block.seg_end_coordinate

    if prev_block is not None:
        num_of_prev_lines = prev_block.num_of_lines
        pre_block_seg_end_coordinate = prev_block.seg_end_coordinate
        prev_end_space_small = (
            abs(prev_block.end_coordinate - pre_block_seg_end_coordinate) < 10
        )
        prev_lines_more_than_one = num_of_prev_lines > 1

        overlap_blocks = (
            context_left_coordinate < prev_block.end_coordinate
            and context_right_coordinate > prev_block.start_coordinate
        )

        # update context_left_coordinate and context_right_coordinate
        if overlap_blocks:
            context_left_coordinate = min(
                prev_block.start_coordinate, context_left_coordinate
            )
            context_right_coordinate = max(
                prev_block.end_coordinate, context_right_coordinate
            )
            prev_end_space_small = (
                abs(context_right_coordinate - pre_block_seg_end_coordinate) < 10
            )
            edge_distance = 0
        else:
            edge_distance = abs(block.start_coordinate - prev_block.end_coordinate)

        current_start_space_small = seg_start_coordinate - context_left_coordinate < 10

        if (
            prev_end_space_small
            and current_start_space_small
            and prev_lines_more_than_one
            and edge_distance < max(prev_block.width, block.width)
        ):
            seg_start_flag = False
    else:
        if seg_start_coordinate - context_left_coordinate < 10:
            seg_start_flag = False

    if context_right_coordinate - seg_end_coordinate < 10:
        seg_end_flag = False

    return seg_start_flag, seg_end_flag