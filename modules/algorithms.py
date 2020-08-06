"""This file contains utility functions for find icon computer vision algorithms."""
from typing import List, Tuple

import cv2
from modules.bounding_box import BoundingBox
import numpy as np
import sklearn.cluster

# experimentally-derived constants for the precision-recall curve
_HIGH_PRECISION_MULTIPLIER = 1.5
_OPTIMAL_ACCURACY_MULTIPLIER = 3
_HIGH_RECALL_MULTIPLIER = 13


def shape_context_distance(icon_contour: np.ndarray,
                           image_contour: np.ndarray) -> float:
  """Calculates the shape context distance bewteen two contours.

  Arguments:
      icon_contour: A list with shape (n, 1, 2).
       Represents the template icon contour
      image_contour: A list with shape (n, 1, 2).
       Represents the image patch contour.
       (Note: function will fail unless the number of channels is 2.)

  Returns:
      float: the shape context distance between the two contours.
  """
  extractor = cv2.createShapeContextDistanceExtractor()
  return extractor.computeDistance(icon_contour, image_contour)


def detect_contours(
    image: np.ndarray,
    use_bilateral_filter: bool,
    find_contour_approx: int = cv2.CHAIN_APPROX_NONE) -> List[List[List[int]]]:
  """Detects the contours in the image.

  Arguments:
      image: Input image, as ndarray.
      use_bilateral_filter: whether to use bilateral filter
       to smooth out edges before Canny edge detection.
      find_contour_approx: the mode taken in by findContours
       regarding approximating contours. The default is no approximation;
       other options include cv2.CHAIN_APPROX_SIMPLE
        and CHAIN_APPROX_TC89_L1 or CHAIN_APPROX_TC89_KCOS

  Returns:
      List of contour groups,each of which is a list of points.
  """
  image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if use_bilateral_filter:
    image_gray = cv2.bilateralFilter(image_gray, 5, 20, 50)
  edges = cv2.Canny(image_gray, 10, 25)
  image_contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                       find_contour_approx)
  return image_contours


def cluster_contours_dbscan(
    image_contours: np.ndarray,
    eps: float = 10,
    min_samples: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  """Group contours using DBSCAN.

  Arguments:
      image_contours: Flattened list of all points in all contours of an image.
      That is: List[List[int]]
      eps: The maximum distance a point can be away to be considered
       within neighborhood of another point. (default: {10})
      min_samples: The number of points needed within a neighborhood
       of a point for it to be a core point. (default: {5})

  Returns:
      Tuple: (List of groups of points each representing its own contour,
       List of corresponding boolean mask of core points).
       The contours and masks are np.ndarrays of List[List[int]]
       and List[bool].
  """
  clusters = sklearn.cluster.DBSCAN(
      eps=eps, min_samples=min_samples).fit(image_contours)
  # a label of -1 means the point was not clustered by DBSCAN - a "noise" point
  n_clusters = len(set(
      clusters.labels_)) - (1 if -1 in clusters.labels_ else 0)
  n_noise = list(clusters.labels_).count(-1)
  print("Estimated number of clusters: %d" % n_clusters)
  print("Estimated number of noise points: %d" % n_noise)
  core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
  core_samples_mask[clusters.core_sample_indices_] = True
  contour_groups = []
  core_samples_mask_groups = []
  for i in range(0, n_clusters):
    contour_group = image_contours[np.argwhere(clusters.labels_ == i)]
    contour_groups.append(np.vstack(contour_group).squeeze())
    core_samples_mask_groups.append(
        np.vstack(
            core_samples_mask[np.argwhere(clusters.labels_ == i)]).squeeze())
  return contour_groups, core_samples_mask_groups


def get_bounding_boxes_from_contours(
    contours: np.ndarray) -> Tuple[List[BoundingBox], List[List[int]]]:
  """Convert a list of contours into a list of corresponding bounding boxes.

  Arguments:
      contours: list of contours (which is a list of x,y points) to convert

  Returns:
      Tuple[List of bounding boxes as BoundingBoxes,
      List of bounding boxes as OpenCV rectangles]
  """
  bboxes = []
  rects = []
  for contour in contours:
    contours_poly = cv2.approxPolyDP(contour, 3, True)
    x, y, w, h = cv2.boundingRect(contours_poly)
    rects.append([x, y, w, h])
    bbox = BoundingBox(x, y, x + w, y + h)
    bboxes.append(bbox)
  return bboxes, rects


def filter_unlikely_bounding_boxes(
    sorted_sc_distances: np.ndarray,
    desired_confidence: float = 0.5) -> Tuple[int, int]:
  """Filter out bounding boxes that are unlikely to occur.

  Unlikeliness is quantified by a desired confidence.

  Arguments:
      sorted_sc_distances: list of shape context distances,in sorted order.
      desired_confidence: The desired confidence for the bounding boxes that
         are returned, from 0 to 1. (default: {0.5})

  Returns:
      Tuple[start index, end index] of the subarray of distances
       (and corresponding bounding boxes) that should be kept.
  """
  start_index = 0
  end_index = 0
  middle_confidence_level = 0.5

  # optimize for accuracy if confidence is not super high or low
  if desired_confidence == middle_confidence_level:
    relative_distance_multiplier = _OPTIMAL_ACCURACY_MULTIPLIER

  # optimize for recall if confidence is low
  elif desired_confidence < middle_confidence_level:
    relative_distance_multiplier = _OPTIMAL_ACCURACY_MULTIPLIER + (
        1 - desired_confidence) * (_HIGH_RECALL_MULTIPLIER -
                                   _OPTIMAL_ACCURACY_MULTIPLIER)

  # optimize for precision if confidence is high
  elif desired_confidence > middle_confidence_level:
    relative_distance_multiplier = _HIGH_PRECISION_MULTIPLIER + (
        1 - desired_confidence) * (_OPTIMAL_ACCURACY_MULTIPLIER -
                                   _HIGH_PRECISION_MULTIPLIER)

  # no absolute distance threshold because we assume that at least one icon
  # is present
  relative_max_dist = relative_distance_multiplier * sorted_sc_distances[
      start_index]

  curr_distance = sorted_sc_distances[end_index]
  # we want end_index to be one more than the index of the last kept bbox
  while curr_distance < relative_max_dist:
    end_index += 1
    if end_index < len(sorted_sc_distances):
      curr_distance = sorted_sc_distances[end_index]
    else:
      break
  return start_index, end_index


def suppress_overlapping_bounding_boxes(
    bboxes: List[BoundingBox],
    rects: List[List[int]],
    confidences: np.ndarray,
    confidence_threshold: float,
    iou_threshold: float = 0.9,
) -> List[BoundingBox]:
  """Returns bounding boxes after filtering through non-max suppression.

  Arguments:
      bboxes: list of BoundingBoxes which we will filter
      rects: list of OpenCV rects corresponding to bboxes
      confidences: confidence scores associated with each contour.
      confidence_threshold: only keep bboxes with
       confidence scores (strictly) above this threshold.
      iou_threshold: two bboxes with an IOU >
       this threshold are considered the same bbox. (default: 0.9)

  Returns:
      List of BoundingBoxes that passed the filter.
  """
  indices = cv2.dnn.NMSBoxes(rects, confidences, confidence_threshold,
                             iou_threshold)
  return [bboxes[i[0]] for i in indices]
