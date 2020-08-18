"""This modules contains functions to streamline optimizations."""

from typing import List

from modules import analysis_util
from modules import benchmark_pipeline
from modules import icon_finder_shape_context
import numpy as np


def dbscan_clustering_optimizer(eps_values: List[float],
                                min_samples: List[int], tfrecord_path: str,
                                multi_instance_icon: bool):
  """Plots recall given different DBSCAN clustering hyperparameters.

  User can check the plots to visually check what the overall trend is to
  determine what the next set of hyperpameters to try next (ie, what direction
  the recall is moving). We make a plot of min samples and recall for each
  epsilon value fixed, and we also make a final plot of epsilon and recall
  using the min sample value that maximizes recall.

  Arguments:
      eps_values: List of dbscan epsilon hyperparameters to try out.
      min_samples: List of dbscan min_sample hyperparameters to try out.
      tfrecord_path: The path to the dataset to run this experiment on.
      multi_instance_icon: Whether the dataset is single-instance
       or multi-instance.
  """
  recall_eps = []
  for eps in eps_values:
    recall_min_samples = []
    for samples in min_samples:
      icon_finder = icon_finder_shape_context.IconFinderShapeContext(
          dbscan_eps=eps, dbscan_min_neighbors=samples)
      benchmark = benchmark_pipeline.BenchmarkPipeline(
          tfrecord_path=tfrecord_path)
      correctness, _, _ = benchmark.evaluate(
          multi_instance_icon=multi_instance_icon,
          icon_finder_object=icon_finder)
      recall_min_samples.append(correctness.recall)
    recall_eps.append(np.max(np.array(min_samples)))
    analysis_util.generate_scatterplot(
        x=min_samples,
        y=recall_min_samples,
        title="Effect of min samples on recall (Eps = %d)" % eps,
        xlabel="Min samples",
        ylabel="Recall",
        output_path="min-samples-%d.png" % eps,
        connect_points=False)
  analysis_util.generate_scatterplot(
      x=eps_values,
      y=recall_eps,
      title="Effect of eps on recall (Min sample = best of " +
      " ".join(map(str, min_samples)),
      xlabel="Epsilon Value",
      ylabel="Recall",
      output_path="best-epsilon-recall.png",
      connect_points=False)


if __name__ == "__main__":
  dbscan_clustering_optimizer(
      eps_values=[7.5, 7.6, 7.7, 7.8],
      min_samples=[2, 3, 4, 5],
      tfrecord_path="datasets/large_single_instance_v2.tfrecord",
      multi_instance_icon=False)
