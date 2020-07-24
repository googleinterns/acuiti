"""This module has an IconFinderShapeContext class for finding bounding boxes.
"""
from typing import List, Tuple

import cv2

from modules import algorithms
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
               nms_iou_threshold: float = 0.9):
    """Initializes the hyperparameters for the shape context icon finder.

    Arguments:
        dbscan_eps: The maximum distance a point can be away to be considered
         within neighborhood of another point by DBSCAN. (default: {10})
        dbscan_min_neighbors: The number of points needed within a neighborhood
         of a point for it to be a core point by DBSCAN. (default: {5})
        sc_max_num_points: The maximum number of points per image patch passed
         into shape context descriptor algorithm; can vary slightly for icon
         (default: {100})
        sc_distance_threshold: The maximum shape context distance between an
         icon and an image patch for the image patch to be under consideration
         (default: {0.3})
        nms_iou_threshold: The maximum IOU between two preliminary bounding
         boxes of image patches before the lower confidence one is discarded by
         non-max-suppression algorithm (default: {0.9})
    """
    self.dbscan_eps = dbscan_eps
    self.dbscan_min_neighbors = dbscan_min_neighbors
    self.sc_max_num_points = sc_max_num_points
    self.sc_distance_threshold = sc_distance_threshold
    self.nms_iou_threshold = nms_iou_threshold

  def _get_nearby_contours_and_distances(
      self, icon_contour: np.ndarray,
      image_contours_clusters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to find the image contours closest to the icon.

    Arguments:
        icon_contour: List of points [x, y] representing the icon's contour.
        More precisely, the type is: List[List[int]]
        image_contours_clusters: List of lists of points [x, y] representing
         each of the image's contour clusters. List[List[List[int]]]

    Returns:
        Tuple: (List of contours that are below the distance threshold
        away from the icon: List[List[int]], List of distances corresponding
        to each contour: List[float])
    """

    nearby_contours = []
    nearby_distances = []

    if icon_contour.shape[0] > self.sc_max_num_points:
      downsampled_icon_contour = icon_contour[np.random.choice(
          icon_contour.shape[0], icon_contour.shape[0] // 2, replace=False), :]
    else:
      downsampled_icon_contour = icon_contour

    for image_contour_cluster in image_contours_clusters:
      icon_contour_3d = np.expand_dims(downsampled_icon_contour,
                                       axis=1)
      if image_contour_cluster.shape[0] > self.sc_max_num_points:
        downsampled_image_contour = image_contour_cluster[
            np.random.choice(image_contour_cluster.shape[0],
                             self.sc_max_num_points,
                             replace=False), :]
      else:
        downsampled_image_contour = image_contour_cluster
      image_contour_3d = np.expand_dims(downsampled_image_contour,
                                        axis=1)
      try:
        distance = algorithms.shape_context_distance(icon_contour_3d,
                                                     image_contour_3d)
        if distance < self.sc_distance_threshold:
          nearby_contours.append(image_contour_cluster)
          nearby_distances.append(distance)
      except cv2.error as e:
        print(e)
        print("These were the icon and image shapes: %s %s" %
              (str(icon_contour_3d.shape), str(image_contour_3d.shape)))
    return np.array(nearby_contours), np.array(nearby_distances)

  def find_icons(self, image: np.ndarray,
                 icon: np.ndarray) -> List[BoundingBox]:
    """Find instances of icon in a given image via shape context descriptor.

    Arguments:
        image: Numpy array representing image
        icon: Numpy array representing icon

    Returns:
        List[BoundingBox] -- Bounding Box for each instance of icon in image.
    """

    image_contours = np.vstack(algorithms.detect_contours(image,
                                                          True)).squeeze()
    icon_contour = np.vstack(algorithms.detect_contours(icon, True)).squeeze()
    image_contours_clusters, _ = algorithms.cluster_contours_dbscan(
        image_contours, self.dbscan_eps, self.dbscan_min_neighbors)

    nearby_contours, nearby_distances = self._get_nearby_contours_and_distances(
        icon_contour, image_contours_clusters)
    sorted_indices = nearby_distances.argsort()
    sorted_contours = nearby_contours[sorted_indices]
    sorted_distances = nearby_distances[sorted_indices]
    print("Minimum distance achieved: %f" % sorted_distances[0])
    # invert distances since we want confidence scores
    bboxes = algorithms.suppress_overlapping_bounding_boxes(
        sorted_contours, 1 / sorted_distances, 1 / self.sc_distance_threshold,
        self.nms_iou_threshold)
    return bboxes
