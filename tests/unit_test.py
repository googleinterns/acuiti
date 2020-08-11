from modules import algorithms
from modules import util
import modules.benchmark_pipeline
from modules.bounding_box import BoundingBox
from modules.confusion_matrix import ConfusionMatrix
from modules.correctness_metrics import CorrectnessMetrics
import numpy as np
import pytest

# ------------------------------------------------------------------------
# ------------------------------- test IOU logic -------------------------
# -------------------------------------------------------------------------
_BOX_A = BoundingBox(20, 20, 29, 29)
_BOX_B = BoundingBox(0, 0, 24, 24)
_BOX_C = BoundingBox(1, 2, 3, 4)
_BOX_D = BoundingBox(2, 3, 4, 5)
_BOX_E = BoundingBox(3, 1, 4, 2)
_BOX_F = BoundingBox(2, 2, 3, 3)
_BOX_G = BoundingBox(2, 1, 3, 2)
# The following test suite tests the IOU calculation for bounding boxes
# in these cases, in order:
# -- partial overlap between boxes (diagonal on upper right)
# -- no overlap between bounding boxes
# -- one box within another box
# -- partial overlap between boxes (diagonal square on upper right)
# -- partial overlap between boxes (exactly one pixel, on upper left)
# -- partial overlap between boxes (an entire top/bottom side)
# -- partial overlap between boxes (an entire left/right side)
# covers: no overlap, 1 box overlap, overlap an entire side, partial overlap
bounding_box_tests = [(_BOX_A, _BOX_B, 25 / 700), (_BOX_A, _BOX_C, 0),
                      (_BOX_B, _BOX_C, 9 / 625), (_BOX_C, _BOX_D, 4 / 14),
                      (_BOX_E, _BOX_F, 1 / 7), (_BOX_F, _BOX_G, 2 / 6),
                      (_BOX_E, _BOX_G, 2 / 6)]


@pytest.mark.parametrize("box_1,box_2,expected", bounding_box_tests)
def test_iou(box_1, box_2, expected):
  assert box_1.calculate_iou(box_2) == expected


# ------------------------------------------------------------------------
# ------- test multi-instance accuracy, precision, and recall calculations
# -------------------------------------------------------------------------

_BOX_LIST_1 = [[_BOX_A, _BOX_B], [_BOX_C, _BOX_D]]
_BOX_LIST_2 = [[_BOX_B, _BOX_A], [_BOX_D, _BOX_C]]
_BOX_LIST_3 = [[_BOX_A], [_BOX_C]]
_BOX_LIST_4 = [[_BOX_E], [_BOX_F]]
_BOX_LIST_5 = [[_BOX_G], [_BOX_G]]
# The following two test suites test the evaluation functionality for
# these cases, in order:
# -- proposed boxes that are all correct but not listed in the same order
#    as gold boxes
# -- extra proposed box present, beyond a correct one
# -- one fewer proposed box present, besides a correct one
# -- no gold or proposed boxes present
# -- no proposed boxes but there are gold boxes present
# -- there are proposed boxes returned but no gold boxes present
# -- proposed boxes that have an IOU with the gold boxes *exactly* at the IOU
#    threshold
# -- proposed boxes that have an IOU with the gold boxes below the IOU threshold
#    in particular, each proposed box is a false positive, and each gold box is
#    a false negative (double penalty)

# test the final accuracy, precision, and recall values
correctness_evaluation_tests = [
    (1, _BOX_LIST_1, _BOX_LIST_2, (CorrectnessMetrics(1, 1, 1), [1, 1])),
    (1, _BOX_LIST_1, _BOX_LIST_3, (CorrectnessMetrics(0.5, 0.5, 1), [0, 0])),
    (1, _BOX_LIST_3, _BOX_LIST_1, (CorrectnessMetrics(0.5, 1, 0.5), [0, 0])),
    (1, [[]], [[]], (CorrectnessMetrics(1, 1, 1), [1])),
    (1, [[], []], _BOX_LIST_3, (CorrectnessMetrics(0, 1, 0), [0, 0])),
    (1, _BOX_LIST_3, [[], []], (CorrectnessMetrics(0, 0, 1), [0, 0])),
    (2 / 6, _BOX_LIST_4, _BOX_LIST_5, (CorrectnessMetrics(1, 1, 1), [1, 1])),
    (1, _BOX_LIST_3, _BOX_LIST_5, (CorrectnessMetrics(0, 0, 0), [0, 0]))
]

# explicitly test the intermediate confusion matrix values
#   specifically, a regression test for a case where there were zero gold
#   boxes but nonzero proposed boxes, and the final accuracy/precison/recall
#   values were correct, but not the intermediate confusion matrix values
#   ((false positive, false negative), (true positive, true negative))
confusion_matrix_tests = [
    (1, _BOX_LIST_1, _BOX_LIST_2, (ConfusionMatrix(0, 0, 4, 0), [1, 1])),
    (1, _BOX_LIST_1, _BOX_LIST_3, (ConfusionMatrix(2, 0, 2, 0), [0, 0])),
    (1, _BOX_LIST_3, _BOX_LIST_1, (ConfusionMatrix(0, 2, 2, 0), [0, 0])),
    (1, [[]], [[]], (ConfusionMatrix(0, 0, 0, 1), [1])),
    (1, [[], []], _BOX_LIST_3, (ConfusionMatrix(0, 2, 0, 0), [0, 0])),
    (1, _BOX_LIST_3, [[], []], (ConfusionMatrix(2, 0, 0, 0), [0, 0])),
    (2 / 6, _BOX_LIST_4, _BOX_LIST_5, (ConfusionMatrix(0, 0, 2, 0), [1, 1])),
    (1, _BOX_LIST_3, _BOX_LIST_5, (ConfusionMatrix(2, 2, 0, 0), [0, 0]))
]


# "expected": CorrectnessMetrics dataclass object (accuracy, precision, recall)
@pytest.mark.parametrize("iou_threshold,proposed_boxes,gold_boxes,expected",
                         correctness_evaluation_tests)
def test_evaluate_proposed_bounding_boxes(iou_threshold, proposed_boxes,
                                          gold_boxes, expected):
  assert util.evaluate_proposed_bounding_boxes(iou_threshold, proposed_boxes,
                                               gold_boxes) == expected


# "expected": num of ((false pos, false neg), (true pos, true neg))
@pytest.mark.parametrize("iou_threshold,proposed_boxes,gold_boxes,expected",
                         confusion_matrix_tests)
def test_get_confusion_matrix(iou_threshold, proposed_boxes, gold_boxes,
                              expected):
  assert util.get_confusion_matrix(iou_threshold, proposed_boxes,
                                   gold_boxes) == expected


# ------------------------------------------------------------------------
# ------------------test shape context algorithm -------------------------
# -------------------------------------------------------------------------
icon_contour_1 = np.array([[0, 0], [0, 4], [4, 0], [4, 4]])
icon_contour_2 = np.array([[0, 0], [0, 7], [7, 0], [7, 7]])
icon_contour_3 = np.array([[0, 0], [0, 6], [4, 0], [4, 6]])
icon_contour_1_3d = np.expand_dims(icon_contour_1, axis=1)
icon_contour_2_3d = np.expand_dims(icon_contour_2, axis=1)
icon_contour_3_3d = np.expand_dims(icon_contour_3, axis=1)

# covers: same shape against itself, expected distance ~0
# (comparison across shapes yields Matrix Operand Error)
shape_context_tests = [
    (icon_contour_2_3d, icon_contour_2_3d, 1e-15),
    (icon_contour_1_3d, icon_contour_1_3d, 1e-15),
    (icon_contour_3_3d, icon_contour_3_3d, 1e-15),
]


@pytest.mark.parametrize("icon_1,icon_2,expected", shape_context_tests)
def test_shape_context(icon_1, icon_2, expected):
  assert algorithms.shape_context_distance(icon_1, icon_2) <= expected


# ------------------------------------------------------------------------
# -------------------------test non-max suppression -----------------------
# -------------------------------------------------------------------------

icon_bbox_1 = BoundingBox(0, 0, 4, 4)
icon_bbox_2 = BoundingBox(0, 0, 7, 7)
icon_bbox_3 = BoundingBox(0, 0, 4, 6)
icon_bbox_4 = BoundingBox(0, 0, 10, 10)
icon_bbox_5 = BoundingBox(0, 0, 9, 10)
icon_rect_1 = [0, 0, 4, 4]
icon_rect_2 = [0, 0, 7, 7]
icon_rect_3 = [0, 0, 4, 6]
icon_rect_4 = [0, 0, 10, 10]
icon_rect_5 = [0, 0, 9, 10]
icon_box_list_1 = [icon_bbox_1, icon_bbox_2, icon_bbox_3]
icon_box_list_2 = [icon_bbox_4, icon_bbox_5]
icon_rect_list_1 = [icon_rect_1, icon_rect_2, icon_rect_3]
icon_rect_list_2 = [icon_rect_4, icon_rect_5]
confidence_1 = [5, 6, 7]
confidence_2 = [5, 4]

# contour list 1: tests varying confidence thresholds given IOUs < nms_threshold
# contour list 2: tests varying confidence thresholds given IOU > nms_threshold
nms_tests = [
    (icon_box_list_1, icon_rect_list_1, confidence_1, 2, 0.9, 3),
    (icon_box_list_1, icon_rect_list_1, confidence_1, 6, 0.9, 1),
    (icon_box_list_2, icon_rect_list_2, confidence_2, 3, 0.89, 1),
    (icon_box_list_2, icon_rect_list_2, confidence_2, 6, 0.9, 0),
]


@pytest.mark.parametrize(
    "bboxes,rects,confidences,confidence_threshold,nms_threshold,expected",
    nms_tests)
def test_get_nms_bounding_boxes(bboxes, rects, confidences,
                                confidence_threshold, nms_threshold, expected):
  assert len(
      algorithms.suppress_overlapping_bounding_boxes(
          bboxes, rects, confidences, confidence_threshold,
          nms_threshold)) == expected


# ------------------------------------------------------------------------
# ---------------------- test upsampling & downsampling -------------------
# -------------------------------------------------------------------------
# The following test suite tests the creation of pointsets of a certain desired
# size in these cases, in order:
# -- number of keypoints is exactly max
# -- number of keypoints is less than min,
#     and there are enough nonkeypoints to get to min
# -- number of keypoints is less than min,
#     but there aren't enough nonkeypoints to get to min
# -- number of keypoints is less than min,
#     but there are no keypoints
# -- number of keypoints is more than max
# -- number of keypoints is exactly min
# -- number of keypoints is less than max and more than min
keypoints_1 = np.full((3, 2), 1)
keypoints_2 = np.full((6, 2), 1)
keypoints_3 = np.full((5, 2), 1)
nonkeypoints_1 = np.full((4, 2), 1)
nonkeypoints_2 = np.full((7, 2), 1)
nonkeypoints_3 = np.full((8, 2), 1)

pointset_tests = [(keypoints_1, 2, 3, nonkeypoints_1, 3),
                  (keypoints_2, 7, 7, nonkeypoints_2, 7),
                  (keypoints_3, 14, 15, nonkeypoints_3, 13),
                  (keypoints_1, 4, 5, None, 3),
                  (keypoints_2, 3, 3, nonkeypoints_2, 3),
                  (keypoints_3, 5, 5, nonkeypoints_3, 5),
                  (keypoints_1, 1, 4, nonkeypoints_1, 3)]


@pytest.mark.parametrize(
    "keypoints,min_points,max_points,nonkeypoints,expected", pointset_tests)
def test_create_pointset(keypoints, min_points, max_points, nonkeypoints,
                         expected):
  assert len(
      algorithms.create_pointset(keypoints,
                                 min_points,
                                 max_points,
                                 nonkeypoints,
                                 random_seed=0)) == expected


# ------------------------------------------------------------------------
# ---------------------- test entire benchmark pipeline -------------------
# -------------------------------------------------------------------------
def test_benchmark():
  find_icon_benchmark = modules.benchmark_pipeline.BenchmarkPipeline()
  correctness, avg_time_secs, avg_memory_mibs = find_icon_benchmark.evaluate()
  assert avg_memory_mibs <= 1000
  assert avg_time_secs <= 60
  assert correctness.accuracy >= 0
  assert correctness.precision >= 0
  assert correctness.recall >= 0
