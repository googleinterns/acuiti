from modules import icon_finder_shape_context
import modules.benchmark_pipeline


# ------------------------------------------------------------------------
# ---------------------- test entire benchmark pipeline -------------------
# -------------------------------------------------------------------------
def test_benchmark():
  find_icon_benchmark = modules.benchmark_pipeline.BenchmarkPipeline()
  correctness, avg_time_secs, avg_memory_mibs = find_icon_benchmark.evaluate(
      icon_finder_object=icon_finder_shape_context.IconFinderShapeContext())
  assert avg_memory_mibs <= 1000
  assert avg_time_secs <= 60
  assert correctness.accuracy >= 0
  assert correctness.precision >= 0
  assert correctness.recall >= 0


def test_single_instance_benchmark():
  find_icon_single_instance = modules.benchmark_pipeline.BenchmarkPipeline(
      tfrecord_path="datasets/benchmark_single_instance.tfrecord")
  correctness, avg_time_secs, avg_memory_mibs = find_icon_single_instance.evaluate(
      icon_finder_object=icon_finder_shape_context.IconFinderShapeContext())
  # current results to prevent any regressions due to algorithm changes
  assert avg_memory_mibs <= 1000
  assert avg_time_secs <= 5
  assert correctness.accuracy >= 0.8
  assert correctness.precision >= 0.8
  assert correctness.recall >= 0.8


def test_multi_instance():
  find_icon_multi_instance = modules.benchmark_pipeline.BenchmarkPipeline(
      tfrecord_path="datasets/benchmark_multi_instance.tfrecord")
  # test responsiveness to different desired levels of confidence (from 0 to 1)
  correctness, _, _ = find_icon_multi_instance.evaluate(
      icon_finder_object=icon_finder_shape_context.IconFinderShapeContext(
          desired_confidence=0.9),
      multi_instance_icon=True)
  assert correctness.precision >= 0.7

  find_icon_multi_instance = modules.benchmark_pipeline.BenchmarkPipeline(
      tfrecord_path="datasets/benchmark_multi_instance.tfrecord")
  correctness, _, _ = find_icon_multi_instance.evaluate(
      icon_finder_object=icon_finder_shape_context.IconFinderShapeContext(
          desired_confidence=0.1),
      multi_instance_icon=True)
  assert correctness.recall >= 0.8
