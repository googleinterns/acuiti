from modules.benchmark_pipeline import BenchmarkPipeline
from modules.bounding_box import BoundingBox
import pytest

box_a = BoundingBox(20, 20, 29, 29)
box_b = BoundingBox(0, 0, 24, 24)
box_c = BoundingBox(1, 2, 3, 4)
box_d = BoundingBox(2, 3, 4, 5)
box_e = BoundingBox(3, 1, 4, 2)
box_f = BoundingBox(2, 2, 3, 3)
box_g = BoundingBox(2, 1, 3, 2)
bounding_box_tests = [(box_a, box_b, 25 / 700), (box_a, box_c, 0),
                      (box_b, box_c, 9 / 625), (box_c, box_d, 4 / 14),
                      (box_e, box_f, 1 / 7), (box_f, box_g, 2 / 6),
                      (box_e, box_g, 2 / 6)]


@pytest.mark.parametrize("box_1,box_2,expected", bounding_box_tests)
def test_iou(box_1, box_2, expected):
  assert BenchmarkPipeline.calculate_iou(box_1, box_2) == expected


def test_benchmark():
  benchmark = BenchmarkPipeline()
  find_icon_time, find_icon_memory = benchmark.find_icons()
  accuracy = benchmark.evaluate()
  assert find_icon_memory <= 1000
  assert find_icon_time <= 120
  assert accuracy >= 0
