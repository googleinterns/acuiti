from modules import algorithms
import modules.benchmark_pipeline
from modules.bounding_box import BoundingBox
import numpy as np
import pytest

box_a = BoundingBox(20, 20, 29, 29)
box_b = BoundingBox(0, 0, 24, 24)
box_c = BoundingBox(1, 2, 3, 4)
box_d = BoundingBox(2, 3, 4, 5)
box_e = BoundingBox(3, 1, 4, 2)
box_f = BoundingBox(2, 2, 3, 3)
box_g = BoundingBox(2, 1, 3, 2)
# covers: no overlap, 1 box overlap, overlap an entire side, partial overlap
bounding_box_tests = [(box_a, box_b, 25 / 700), (box_a, box_c, 0),
                      (box_b, box_c, 9 / 625), (box_c, box_d, 4 / 14),
                      (box_e, box_f, 1 / 7), (box_f, box_g, 2 / 6),
                      (box_e, box_g, 2 / 6)]


@pytest.mark.parametrize("box_1,box_2,expected", bounding_box_tests)
def test_iou(box_1, box_2, expected):
  assert box_1.calculate_iou(box_2) == expected


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
icon_bbox_list_1 = [icon_bbox_1, icon_bbox_2, icon_bbox_3]
icon_bbox_list_2 = [icon_bbox_4, icon_bbox_5]
icon_rect_list_1 = [icon_rect_1, icon_rect_2, icon_rect_3]
icon_rect_list_2 = [icon_rect_4, icon_rect_5]
confidence_1 = [5, 6, 7]
confidence_2 = [5, 4]

# contour list 1: tests varying confidence thresholds given IOUs < nms_threshold
# contour list 2: tests varying confidence thresholds given IOU > nms_threshold
nms_tests = [
    (icon_bbox_list_1, icon_rect_list_1, confidence_1, 2, 0.9, 3),
    (icon_bbox_list_1, icon_rect_list_1, confidence_1, 6, 0.9, 1),
    (icon_bbox_list_2, icon_rect_list_2, confidence_2, 3, 0.89, 1),
    (icon_bbox_list_2, icon_rect_list_2, confidence_2, 6, 0.9, 0),
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


def test_benchmark():
  find_icon_benchmark = modules.benchmark_pipeline.BenchmarkPipeline()
  accuracy, avg_time_secs, avg_memory_mibs = find_icon_benchmark.evaluate()
  assert avg_memory_mibs <= 1000
  assert avg_time_secs <= 60
  assert accuracy >= 0
