from modules import util
import modules.benchmark_pipeline
from modules.bounding_box import BoundingBox
import pytest

_BOX_A = BoundingBox(20, 20, 29, 29)
_BOX_B = BoundingBox(0, 0, 24, 24)
_BOX_C = BoundingBox(1, 2, 3, 4)
_BOX_D = BoundingBox(2, 3, 4, 5)
_BOX_E = BoundingBox(3, 1, 4, 2)
_BOX_F = BoundingBox(2, 2, 3, 3)
_BOX_G = BoundingBox(2, 1, 3, 2)
# covers: no overlap, 1 box overlap, overlap an entire side, partial overlap
bounding_box_tests = [(_BOX_A, _BOX_B, 25 / 700), (_BOX_A, _BOX_C, 0),
                      (_BOX_B, _BOX_C, 9 / 625), (_BOX_C, _BOX_D, 4 / 14),
                      (_BOX_E, _BOX_F, 1 / 7), (_BOX_F, _BOX_G, 2 / 6),
                      (_BOX_E, _BOX_G, 2 / 6)]


@pytest.mark.parametrize("box_1,box_2,expected", bounding_box_tests)
def test_iou(box_1, box_2, expected):
  assert box_1.calculate_iou(box_2) == expected


box_list_1 = [[_BOX_A, _BOX_B], [_BOX_C, _BOX_D]]
box_list_2 = [[_BOX_B, _BOX_A], [_BOX_D, _BOX_C]]
box_list_3 = [[_BOX_A], [_BOX_C]]
box_list_4 = [[_BOX_E], [_BOX_F]]
box_list_5 = [[_BOX_G], [_BOX_G]]
# Tests proper matching of proposed box to gold box, IOU thresholds
# includes double penalty for false positive and false negative if
# a proposed box doesn't match the IOU threshold and there exists a gold box
# Also includes empty edge cases
evaluation_tests = [(1, box_list_1, box_list_2, (1, 1, 1)),
                    (1, box_list_1, box_list_3, (0.5, 0.5, 1)),
                    (1, box_list_3, box_list_1, (0.5, 1, 0.5)),
                    (1, [[]], [[]], (1, 1, 1)),
                    (1, [[], []], box_list_3, (0, 1, 0)),
                    (1, box_list_3, [[], []], (0, 0, 1)),
                    (2 / 6, box_list_4, box_list_5, (1, 1, 1)),
                    (1, box_list_3, box_list_5, (0, 0, 0))]


# expected is a tuple: (accuracy, precision, recall)
@pytest.mark.parametrize("iou_threshold,proposed_boxes,gold_boxes,expected",
                         evaluation_tests)
def test_evaluate_proposed_bounding_boxes(iou_threshold, proposed_boxes,
                                          gold_boxes, expected):
  assert util.evaluate_proposed_bounding_boxes(iou_threshold, proposed_boxes,
                                               gold_boxes) == expected


def test_benchmark():
  find_icon_benchmark = modules.benchmark_pipeline.BenchmarkPipeline()
  accuracy, precision, recall, avg_time_secs, avg_memory_mibs = find_icon_benchmark.evaluate(
  )
  assert avg_memory_mibs <= 1000
  assert avg_time_secs <= 60
  assert accuracy >= 0
  assert precision >= 0
  assert recall >= 0
