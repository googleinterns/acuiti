"""This module has an IconFinderShapeContext class for finding bounding boxes.
"""
import multiprocessing  # pytype: disable=pyi-error
from typing import List, Tuple

import cv2

from modules import algorithms
from modules import clustering_algorithms
from modules.bounding_box import BoundingBox
import modules.icon_finder
import numpy as np


class IconFinderShapeContext(modules.icon_finder.IconFinder):  # pytype: disable=module-attr
  """This class generates bounding boxes via Shape Context Descriptors."""

  def __init__(self,
               clusterer: clustering_algorithms.
               SklearnClusterer = clustering_algorithms.DBSCANClusterer(),
               desired_confidence: float = 0.5,
               sc_min_num_points: int = 90,
               sc_max_num_points: int = 90,
               sc_distance_threshold: float = 1,
               nms_iou_threshold: float = 0.9):
    """Initializes the hyperparameters for the shape context icon finder.

    Arguments:
        clusterer: A clusterer object that inherits from SklearnClusterer (ie,
        is a wrapper for one of Sklearn's clustering algorithm objects)
        desired_confidence: The desired confidence for the bounding boxes that
         are returned, from 0 to 1. (default: {0.5})
        sc_min_num_points: The *desired* minimum number of points per image
         patch passed into shape context descriptor algorithm, if possible.
         Also applies to template icon. (default: {90})
        sc_max_num_points: The maximum number of points per image patch passed
         into shape context descriptor algorithm. Also applies to template icon.
         (default: {90})
        sc_distance_threshold: The maximum shape context distance between an
         icon and an image patch for the image patch to be under consideration
         (default: {1})
        nms_iou_threshold: The maximum IOU between two preliminary bounding
         boxes of image patches before the lower confidence one is discarded by
         non-max-suppression algorithm (default: {0.9})
    """
    assert isinstance(
        clusterer, clustering_algorithms.SklearnClusterer
    ), "Clusterer passed in must be an instance of SklearnClusterer"
    self.clusterer = clusterer.get_clusterer()
    self.desired_confidence = desired_confidence
    self.sc_min_num_points = sc_min_num_points
    self.sc_max_num_points = sc_max_num_points
    self.sc_distance_threshold = sc_distance_threshold
    self.nms_iou_threshold = nms_iou_threshold

  def _get_distance(self, icon_contour_3d: np.ndarray,
                    image_contour_3d: np.ndarray) -> Optional[Tuple]:
    """Calculate distance between icon and image contour.

    Arguments:
        icon_contour_3d: icon contour in shape context's format
        image_contour_3d: image contour in shape context's format
        (n, 1, 2)

    Returns:
        (distance, image_contour_3d), or None if there was an exception
    """
    try:
      distance = algorithms.shape_context_distance(icon_contour_3d,
                                                   image_contour_3d)
      if distance < self.sc_distance_threshold:
        return (image_contour_3d, distance)
    except cv2.error as e:
      print(e)
      print("These were the icon and image shapes: %s %s" %
            (str(icon_contour_3d.shape), str(image_contour_3d.shape)))

  def _get_similar_contours(
      self, icon_contour_keypoints: np.ndarray,
      icon_contour_nonkeypoints: np.ndarray,
      image_contour_clusters_keypoints: np.ndarray,
      image_contour_clusters_nonkeypoints: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to find the image contours closest to the icon.

    Arguments:
        icon_contour_keypoints: List of points [x, y]
        representing the icon's contour's keypoints. Type: List[List[int]]
        icon_contour_nonkeypoints: List of points [x, y]
        representing the icon's contour's nonkeypoints. Type: List[List[int]]
        image_contour_clusters_keypoints: List of lists of points
         [x, y] representing each of the image's contour clusters' keypoints.
         List[List[List[int]]]
        image_contour_clusters_nonkeypoints: List of lists of points
         [x, y] representing each of the image's contour clusters' nonkeypoints.
         List[List[List[int]]]

    Returns:
        Tuple: (List of contours that are below the distance threshold
        away from the icon: List[List[int]], List of distances corresponding
        to each contour: List[float])
    """

    icon_pointset = algorithms.resize_pointset(icon_contour_keypoints,
                                               self.sc_min_num_points,
                                               self.sc_max_num_points,
                                               icon_contour_nonkeypoints)
    # expand the 1st dimension so that the shape is (n, 1, 2),
    # which is what shape context algorithm wants
    icon_contour_3d = np.expand_dims(icon_pointset, axis=1)

    # uses as many processes as available CPUs
    pool = multiprocessing.Pool(None)
    contours_and_distances = []
    for cluster_keypoints, cluster_nonkeypoints in zip(
        image_contour_clusters_keypoints, image_contour_clusters_nonkeypoints):
      cluster_pointset = algorithms.resize_pointset(cluster_keypoints,
                                                    self.sc_min_num_points,
                                                    self.sc_max_num_points,
                                                    cluster_nonkeypoints)

      # expand the 1st dimension so that the shape is (n, 1, 2),
      # which is what shape context algorithm wants
      image_contour_3d = np.expand_dims(cluster_pointset, axis=1)
      pool.apply_async(self._get_distance,
                       args=(
                           icon_contour_3d,
                           image_contour_3d,
                       ),
                       callback=contours_and_distances.append)

    pool.close()
    pool.join()
    nearby_contours, nearby_distances = zip(
        *list(filter(None, contours_and_distances)))
    return np.array(nearby_contours), np.array(nearby_distances)

  def find_icons(
      self, image: np.ndarray, icon: np.ndarray
  ) -> Tuple[List[BoundingBox], List[np.ndarray], List[np.ndarray]]:
    """Find instances of icon in a given image via shape context descriptor.

    Arguments:
        image: Numpy array representing image
        icon: Numpy array representing icon

    Returns:
        Tuple(list of Bounding Box for each instance of icon in image,
        list of clusters of contours detected in the image to visually evaluate
        how well contour clustering worked, list of booleans representing
        whether each image had zero false positives and false negatives)
    """
    # get icon keypoints and nonkeypoints (using all points will hurt accuracy)
    icon_contour_keypoints = np.vstack(
        algorithms.detect_contours(icon, True,
                                   cv2.CHAIN_APPROX_SIMPLE)).squeeze()
    icon_contour_all = np.vstack(algorithms.detect_contours(icon,
                                                            True)).squeeze()
    icon_contour_keypoints_set = set(map(tuple, icon_contour_keypoints))
    icon_contour_nonkeypoints = np.array([
        point for point in icon_contour_all
        if tuple(point) not in icon_contour_keypoints_set
    ])

    # cluster image contours using all points
    image_contours = np.vstack(algorithms.detect_contours(image,
                                                          True)).squeeze()

    image_contours_clusters = algorithms.cluster_contours(
        self.clusterer, image_contours)

    # filter out nonkeypoints from image contour clusters
    image_contours_keypoints = np.vstack(
        algorithms.detect_contours(image, True,
                                   cv2.CHAIN_APPROX_SIMPLE)).squeeze()
    image_contours_keypoints = set(map(tuple, image_contours_keypoints))

    image_contours_clusters_keypoints = []
    image_contours_clusters_nonkeypoints = []
    # go through each cluster, identify which are keypoints and nonkeypoints
    for cluster in image_contours_clusters:
      keypoint_cluster = []
      nonkeypoint_cluster = []
      # separate the keypoints from non keypoints into different clusters
      for point in cluster:
        if tuple(point) in image_contours_keypoints:
          keypoint_cluster.append(point)
        else:
          nonkeypoint_cluster.append(point)
      image_contours_clusters_keypoints.append(np.array(keypoint_cluster))
      image_contours_clusters_nonkeypoints.append(
          np.array(nonkeypoint_cluster))

    # get nearby contours by using keypoint information
    nearby_contours, nearby_distances = self._get_similar_contours(
        icon_contour_keypoints, icon_contour_nonkeypoints,
        np.array(image_contours_clusters_keypoints),
        np.array(image_contours_clusters_nonkeypoints))
    sorted_indices = nearby_distances.argsort()
    sorted_contours = nearby_contours[sorted_indices]
    sorted_distances = nearby_distances[sorted_indices]
    print("Minimum distance achieved: %f" % sorted_distances[0])
    distance_threshold = algorithms.get_distance_threshold(
        sorted_distances, desired_confidence=self.desired_confidence)
    end_index = np.searchsorted(sorted_distances,
                                distance_threshold,
                                side="right")
    sorted_contours = sorted_contours[0:end_index]
    sorted_distances = sorted_distances[0:end_index]
    bboxes, rects = algorithms.get_bounding_boxes_from_contours(
        sorted_contours)
    # invert distances since we want confidence scores
    bboxes = algorithms.suppress_overlapping_bounding_boxes(
        bboxes, rects, 1 / sorted_distances, 1 / self.sc_distance_threshold,
        self.nms_iou_threshold)
    icon_bbox, _ = algorithms.get_bounding_boxes_from_contours(
        np.array([icon_contour_keypoints]))
    bboxes = algorithms.standardize_bounding_boxes_padding(
        bboxes, icon_bbox[0], icon, image)
    return bboxes, image_contours_clusters_keypoints, icon_contour_keypoints
