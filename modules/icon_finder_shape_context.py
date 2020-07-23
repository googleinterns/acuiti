"""This module has an IconFinderShapeContext class for finding bounding boxes.
"""
from typing import List, Tuple

import cv2

from modules import algo_util
from modules.bounding_box import BoundingBox
import modules.icon_finder
import numpy as np


class IconFinderShapeContext(modules.icon_finder.IconFinder):  # pytype: disable=module-attr
  """This class generates bounding boxes via Shape Context Descriptors."""

  def __init__(self,
               dbscan_eps: float = 10,
               dbscan_min_neighbors: int = 5,
               sc_max_num_points: int = 100,
               sc_distance_threshold: float = 0.3,
               nms_threshold: float = 0.9):
    self.dbscan_eps = dbscan_eps
    self.dbscan_min_neighbors = dbscan_min_neighbors
    self.sc_max_num_points = sc_max_num_points
    self.sc_distance_threshold = sc_distance_threshold
    self.nms_threshold = nms_threshold

  def _get_min_distance_contours(
      self, icon_contour: np.ndarray,
      img_contours_clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to find the image contours closest to the icon.

    Arguments:
        icon_contour: List of points [x, y] representing the icon's contour.
        More precisely, the type is: List[List[int]]
        img_contours_clusters: List of lists of points [x, y] representing
         each of the image's contour clusters. List[List[List[int]]]

    Returns:
        List of contours that are below the distance threshold
        away from the icon: List[List[int]];
        List of distances corresponding to each contour: List[float]
    """

    min_distance_contours = []
    min_distances = []

    if len(icon_contour) > self.sc_max_num_points:
      downsampled_icon_contour = icon_contour[np.random.choice(
          icon_contour.shape[0], len(icon_contour) // 2, replace=False), :]
    else:
      downsampled_icon_contour = icon_contour

    for img_contour_cluster in img_contours_clusters:
      icon_contour_3d = np.expand_dims(np.array(downsampled_icon_contour),
                                       axis=1)
      if len(img_contour_cluster) > self.sc_max_num_points:
        downsampled_image_contour = img_contour_cluster[
            np.random.choice(img_contour_cluster.shape[0],
                             self.sc_max_num_points,
                             replace=False), :]
      else:
        downsampled_image_contour = img_contour_cluster
      image_contour_3d = np.expand_dims(np.array(downsampled_image_contour),
                                        axis=1)
      try:
        distance = algo_util.shape_context_descriptor(icon_contour_3d,
                                                      image_contour_3d)
        if distance < self.sc_distance_threshold:
          min_distance_contours.append(img_contour_cluster)
          min_distances.append(distance)
      except cv2.error as e:
        print(e)
        print("These were the icon and image shapes: %s %s" %
              (str(icon_contour_3d.shape), str(image_contour_3d.shape)))
    return np.array(min_distance_contours), np.array(min_distances)

  def find_icons(self, image: np.ndarray,
                 icon: np.ndarray) -> List[BoundingBox]:
    """Find instances of icon in a given image via shape context descriptor.

    Arguments:
        image: Numpy array representing image
        icon: Numpy array representing icon

    Returns:
        List[BoundingBox] -- Bounding Box for each instance of icon in image.
    """

    img_contours = np.vstack(algo_util.detect_contours(image, True)).squeeze()
    icon_contour = np.vstack(algo_util.detect_contours(icon, True)).squeeze()
    img_contours_clusters, _ = algo_util.cluster_contours_dbscan(
        img_contours, self.dbscan_eps, self.dbscan_min_neighbors)

    min_distance_contours, min_distances = self._get_min_distance_contours(
        icon_contour, img_contours_clusters)
    sorted_contours = min_distance_contours[min_distances.argsort()]
    sorted_distances = np.sort(min_distances)
    print("Minimum distance achieved: %f" % sorted_distances[0])
    # invert distances since we want confidence scores
    bboxes = algo_util.get_nms_bounding_boxes(
        sorted_contours, 1 / sorted_distances,
        1 / self.sc_distance_threshold, self.nms_threshold)
    return bboxes
