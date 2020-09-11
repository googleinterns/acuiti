"""This module contains classes for clustering algorithms.
"""
import abc

import hdbscan
import sklearn
import sklearn.cluster


class SklearnClusterer(abc.ABC):
  """SklearnClusterer is the base class for clustering classes.

  Attributes:
    clusterer: object of type sklearn.base.ClusterMixin. All
     sklearn.base.ClusterMixin objects follow the same API
     (ie, you can call .fit(X) on them).
  """
  clusterer: sklearn.base.ClusterMixin

  def get_clusterer(self):
    return self.clusterer


class DBSCANClusterer(SklearnClusterer):
  """Wrapper class for Sklearn's DBSCAN clusterer.

  Default values are optimized for the shape context algorithm
  on our dataset. The only two parameters of note that we have
  changed from the default are eps and min_samples. Here is the
  corresponding sklearn documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
  """

  def __init__(self, eps: float = 7.5, min_samples: int = 2, **kwargs):
    self.clusterer = sklearn.cluster.DBSCAN(eps=eps,
                                            min_samples=min_samples,
                                            **kwargs)


class AgglomerativeClusterer(SklearnClusterer):
  """Wrapper class for Sklearn's Agglomerative clusterer.

  Default values are optimized for the shape context algorithm
  on our dataset. The only parameters we have changed the
  defaults for are n_clusters and compute_full_tree. Here is the
  corresponding sklearn documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
  """

  def __init__(self,
               n_clusters: int = 45,
               compute_full_tree: bool = False,
               **kwargs):
    self.clusterer = sklearn.cluster.AgglomerativeClustering(
        n_clusters=n_clusters, compute_full_tree=compute_full_tree, **kwargs)


class OPTICSClusterer(SklearnClusterer):
  """Wrapper class for Sklearn's oPTICS clusterer.

  Default values are optimized for the shape context algorithm
  on our dataset. The parameters we have changed from default
  are min_samples, max_eps, min_cluster_size, and n_jobs. Here
  is the corresponding sklearn documentation:
  https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html
  """

  def __init__(self,
               min_samples: int = 40,
               max_eps: float = 7.5,
               min_cluster_size: int = 50,
               n_jobs: int = -1,
               **kwargs):
    self.clusterer = sklearn.cluster.OPTICS(min_samples=min_samples,
                                            max_eps=max_eps,
                                            min_cluster_size=min_cluster_size,
                                            n_jobs=n_jobs,
                                            **kwargs)


class HDBSCANClusterer(SklearnClusterer):
  """Wrapper class for the HDBSCAN clusterer (which follows sklearn's API).

  Default values are optimized for the shape context algorithm
  on our dataset. The docs for HDBSCAN note that only the parameters
  min_cluster_size, min_samples, and cluster_selection_epsilon should be
  tweaked. These are the only three values that we have changed/optimized.
  Here is the corresponding documentation:
  https://hdbscan.readthedocs.io/en/latest/api.html
  """

  def __init__(self,
               min_cluster_size: int = 11,
               min_samples: int = 15,
               cluster_selection_epsilon: float = 7.5,
               **kwargs):
    self.clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_epsilon=cluster_selection_epsilon,
        **kwargs)
