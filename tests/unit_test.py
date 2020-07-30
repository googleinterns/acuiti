from modules import util
import modules.benchmark_pipeline
from modules.bounding_box import BoundingBox
from modules.correctness_metrics import CorrectnessMetrics
import pytest

_BOX_A = BoundingBox(20, 20, 29, 29)
_BOX_B = BoundingBox(0, 0, 24, 24)
_BOX_C = BoundingBox(1, 2, 3, 4)
_BOX_D = BoundingBox(2, 3, 4, 5)
_BOX_E = BoundingBox(3, 1, 4, 2)
_BOX_F = BoundingBox(2, 2, 3, 3)
_BOX_G = BoundingBox(2, 1, 3, 2)
# The following two test suites test the IOU calculation for bounding boxes
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


box_list_1 = [[_BOX_A, _BOX_B], [_BOX_C, _BOX_D]]
box_list_2 = [[_BOX_B, _BOX_A], [_BOX_D, _BOX_C]]
box_list_3 = [[_BOX_A], [_BOX_C]]
box_list_4 = [[_BOX_E], [_BOX_F]]
box_list_5 = [[_BOX_G], [_BOX_G]]
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
    (1, box_list_1, box_list_2, CorrectnessMetrics(1, 1, 1)),
    (1, box_list_1, box_list_3, CorrectnessMetrics(0.5, 0.5, 1)),
    (1, box_list_3, box_list_1, CorrectnessMetrics(0.5, 1, 0.5)),
    (1, [[]], [[]], CorrectnessMetrics(1, 1, 1)),
    (1, [[], []], box_list_3, CorrectnessMetrics(0, 1, 0)),
    (1, box_list_3, [[], []], CorrectnessMetrics(0, 0, 1)),
    (2 / 6, box_list_4, box_list_5, CorrectnessMetrics(1, 1, 1)),
    (1, box_list_3, box_list_5, CorrectnessMetrics(0, 0, 0))
]

# explicitly test the intermediate confusion matrix values
#   specifically, a regression test for a case where there were zero gold
#   boxes but nonzero proposed boxes, and the final accuracy/precison/recall
#   values were correct, but not the intermediate confusion matrix values
#   ((false positive, false negative), (true positive, true negative))
confusion_matrix_tests = [(1, box_list_1, box_list_2, ((0, 0), (4, 0))),
                          (1, box_list_1, box_list_3, ((2, 0), (2, 0))),
                          (1, box_list_3, box_list_1, ((0, 2), (2, 0))),
                          (1, [[]], [[]], ((0, 0), (0, 1))),
                          (1, [[], []], box_list_3, ((0, 2), (0, 0))),
                          (1, box_list_3, [[], []], ((2, 0), (0, 0))),
                          (2 / 6, box_list_4, box_list_5, ((0, 0), (2, 0))),
                          (1, box_list_3, box_list_5, ((2, 2), (0, 0)))]


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


def test_benchmark():
  find_icon_benchmark = modules.benchmark_pipeline.BenchmarkPipeline()
  correctness, avg_time_secs, avg_memory_mibs = find_icon_benchmark.evaluate()
  assert avg_memory_mibs <= 1000
  assert avg_time_secs <= 60
  assert correctness.accuracy >= 0
  assert correctness.precision >= 0
  assert correctness.recall >= 0
