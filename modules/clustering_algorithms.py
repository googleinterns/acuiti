"""This module contains classes for clustering algorithms.
"""
import abc
from typing import Any

import hdbscan
import sklearn.cluster


class SklearnClusterer(abc.ABC):
  """SklearnClusterer is the base class for clustering classes.

  Attributes:
    clusterer: object of type sklearn.cluster. All sklearn.cluster objects
     follow the same API (ie, you can call .fit(X) on them).
  """
  clusterer: sklearn.cluster

  def get_clusterer(self):
    return self.clusterer


class DBSCANClusterer(SklearnClusterer):
  """Wrapper class for Sklearn's DBSCAN clusterer.

  Default values are optimized for the shape context algorithm
  on our dataset. The only two parameters of note that we have
  changed from the default are eps and min_samples.
  """

  def __init__(self,
               eps: float = 7.5,
               min_samples: int = 2,
               metric: str = "euclidean",
               metric_params: Any = None,
               algorithm: str = "auto",
               leaf_size: int = 30,
               p: float = None,
               n_jobs: int = None):
    self.clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                            min_samples=min_samples,
                                            metric=metric,
                                            metric_params=metric_params,
                                            algorithm=algorithm,
                                            leaf_size=leaf_size,
                                            p=p,
                                            n_jobs=n_jobs)


class AgglomerativeClusterer(SklearnClusterer):
  """Wrapper class for Sklearn's Agglomerative clusterer.

  Default values are optimized for the shape context algorithm
  on our dataset. The only parameters we have changed the
  defaults for are n_clusters and compute_full_tree.
  """

  def __init__(self,
               n_clusters: int = 45,
               compute_full_tree: bool = False,
               memory: Any = None,
               connectivity: Any = None,
               linkage: str = "ward",
               distance_threshold: float = None):
    self.clusterer = sklearn.cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        compute_full_tree=compute_full_tree,
        memory=memory,
        connectivity=connectivity,
        linkage=linkage,
        distance_threshold=distance_threshold)


class OPTICSClusterer(SklearnClusterer):
  """Wrapper class for Sklearn's oPTICS clusterer.

  Default values are optimized for the shape context algorithm
  on our dataset. The parameters we have changed from default
  are min_samples, max_eps, min_cluster_size, and n_jobs.
  """

  def __init__(self,
               min_samples: int = 40,
               max_eps: float = 7.5,
               min_cluster_size: int = 50,
               n_jobs: int = -1,
               metric: Any = "minkowski",
               p: int = 2,
               metric_params: Any = None,
               cluster_method: str = "xi",
               eps: float = None,
               xi: float = 0.05,
               predecessor_correction: bool = True,
               algorithm: str = "auto",
               leaf_size: int = 30):
    self.clusterer = sklearn.cluster.OPTICS(
        min_samples=min_samples,
        max_eps=max_eps,
        min_cluster_size=min_cluster_size,
        n_jobs=n_jobs,
        metric=metric,
        p=p,
        metric_params=metric_params,
        cluster_method=cluster_method,
        eps=eps,
        xi=xi,
        predecessor_correction=predecessor_correction,
        algorithm=algorithm,
        leaf_size=leaf_size)


class HDBSCANClusterer(SklearnClusterer):
  """Wrapper class for the HDBSCAN clusterer (which follows sklearn's API).

  Default values are optimized for the shape context algorithm
  on our dataset. The docs for HDBSCAN note that only the parameters
  min_cluster_size, min_samples, and cluster_selection_epsilon should be
  tweaked. These are the only three values that we have changed/optimized.
  """

  def __init__(self,
               min_cluster_size: int = 11,
               min_samples: int = 15,
               cluster_selection_epsilon: float = 7.5,
               metric: Any = "euclidean",
               p: int = None,
               alpha: float = 1.0,
               algorithm: str = "best",
               leaf_size: int = 40,
               memory: Any = None,
               approx_min_span_tree: bool = True,
               gen_min_span_tree: bool = False,
               core_dist_n_jobs: int = 4,
               cluster_selection_method: str = "eom",
               allow_single_cluster: bool = False,
               prediction_data: bool = False,
               match_reference_implementation: bool = False):
    self.clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        metric=metric,
        p=p,
        alpha=alpha,
        algorithm=algorithm,
        leaf_size=leaf_size,
        memory=memory,
        approx_min_span_tree=approx_min_span_tree,
        gen_min_span_tree=gen_min_span_tree,
        core_dist_n_jobs=core_dist_n_jobs,
        cluster_selection_method=cluster_selection_method,
        allow_single_cluster=allow_single_cluster,
        prediction_data=prediction_data,
        match_reference_implementation=match_reference_implementation)
