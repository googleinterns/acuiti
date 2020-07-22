"""This module has an IconFinderRandom class for randomly finding bounding boxes.
"""
from typing import List

import cv2

from modules import algo_util
from modules.bounding_box import BoundingBox
import modules.icon_finder
import numpy as np


class IconFinderShapeContext(modules.icon_finder.IconFinder):  # pytype: disable=module-attr
  """This class generates bounding boxes via Shape Context Descriptors."""

  def _get_min_distance_contour(
      self, icon_contour: List[List[int]],
      img_contours_clusters: List[List[List[int]]]) -> List[List[int]]:
    """Helper function to find the image contour closest to the icon.

    Arguments:
        icon_contour: List of points representing the icon's contour.
        img_contours_clusters: List of groups of points representing
         each of the image's contour clusters.

    Returns:
        List of points (x,y) representing the contour with the
         minimal distance from the icon.
    """
    min_distance = 10000
    min_distance_contour = None
    max_num_points = 100
    if len(icon_contour) > max_num_points:
      downsampled_icon_contour = icon_contour[np.random.choice(
          icon_contour.shape[0], len(icon_contour) // 2, replace=False), :]
    else:
      downsampled_icon_contour = icon_contour

    for img_contour_cluster in img_contours_clusters:
      icon_contour_3d = np.array(
          [downsampled_icon_contour, downsampled_icon_contour])
      if len(img_contour_cluster) > max_num_points:
        downsampled_image_contour = img_contour_cluster[np.random.choice(
            img_contour_cluster.shape[0], max_num_points, replace=False), :]
      else:
        downsampled_image_contour = img_contour_cluster
      image_contour_3d = np.array(
          [downsampled_image_contour, downsampled_image_contour])
      distance = algo_util.shape_context_descriptor(icon_contour_3d,
                                                    image_contour_3d)
      print(distance)
      if distance < min_distance:
        min_distance = distance
        min_distance_contour = img_contour_cluster
    return min_distance_contour

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
    img_contours_clusters, _ = algo_util.cluster_contours_dbscan(img_contours)

    min_distance_contour = self._get_min_distance_contour(
        icon_contour, img_contours_clusters)

    contours_poly = cv2.approxPolyDP(min_distance_contour, 3, True)
    bound_rect = cv2.boundingRect(contours_poly)
    bbox = BoundingBox(bound_rect[0], bound_rect[1],
                       bound_rect[0] + bound_rect[2],
                       bound_rect[1] + bound_rect[3])
    return [bbox]
