"""This file contains utility functions for find icon computer vision algorithms.

The algorithms here generally correspond to what is used in icon_finders:
- Shape context distance
- Contour Detection
- Contour Clustering
- Pointset Resizing
- Contours to Bounding Boxes
- Distance thresholding
- Suppress overlapping bounding boxes
"""
from typing import List, Tuple

import cv2
from modules.bounding_box import BoundingBox
import numpy as np
import sklearn.cluster

# experimentally-derived constants for the precision-recall curve
_HIGH_PRECISION_MULTIPLIER = 1.5
_OPTIMAL_ACCURACY_MULTIPLIER = 3
_HIGH_RECALL_MULTIPLIER = 13
# middle confidence level where confidence is from 0 to 1 (this is constant)
_MIDDLE_CONFIDENCE_LEVEL = 0.5


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


def resize_pointset(keypoints: np.ndarray,
                    min_points: int,
                    max_points: int,
                    nonkeypoints: np.ndarray = None,
                    random_seed: int = 0) -> np.ndarray:
  """Resize pointset to be within [min, max] whenever possible.

  If there are enough keypoints and nonkeypoints, the resulting pointset
  will be at least min_points large. In all cases, the resulting pointset
  will be less than or equal to max_points in size. Nonkeypoints are used
  to bring the size up to min_points if needed (upsampling). Keypoints are only
  downsampled if the current size is greater than max_points.

  Arguments:
      keypoints: an array of [x, y] points representing keypoints
      min_points: the minimum desired number of points we want in pointset
      max_points: the upper limit of points we want in point set
      nonkeypoints: an optional array of [x, y] points representing nonkeypoints
        If provided, these points will be used to bring the pointset size up to
        min_points as much as possible. (Default: None)
      random_seed: the random seed to use for numpy's PRG functions.
       (Default: 0)

  Returns:
      np.ndarray: resulting pointset after possibly upsampling or downsampling.
  """
  # set the random seed *locally*
  random_state = np.random.RandomState(random_seed)
  num_keypoints = 0
  if keypoints is not None:
    num_keypoints = keypoints.shape[0]
  num_nonkeypoints = 0
  if nonkeypoints is not None:
    num_nonkeypoints = nonkeypoints.shape[0]
  pointset = keypoints

  if num_keypoints > max_points:
    # downsample as few keypoints as possible
    pointset = keypoints[
        random_state.choice(num_keypoints, max_points, replace=False), :]

  elif num_keypoints < min_points:
    if num_nonkeypoints:
      if num_keypoints + num_nonkeypoints <= min_points:
        selected_nonkeypoints = nonkeypoints
      else:
        # upsample as few nonkeypoints as possible
        selected_nonkeypoints = nonkeypoints[random_state.choice(
            num_nonkeypoints, min_points - num_keypoints, replace=False), :]
      pointset = np.concatenate((keypoints, selected_nonkeypoints))

  return pointset


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


def get_distance_threshold(
    sc_distances: np.ndarray,
    desired_confidence: float = _MIDDLE_CONFIDENCE_LEVEL) -> float:
  """Return distance threshold that can be used to filter out bounding boxes.

  At a desired confidence of 0, we favor recall the most, whereas at a desired
  confidence of 1, we favor precision the most. In general, the higher the
  confidence, the tighter the distance threshold must be for distances
  to be kept. The lower the confidence, the looser (higher) the distance
  threshold can be. This distance threshold is calculated based off the
  desired confidence, and is a multiple of the minimum distance. Low confidence
  (high recall) means a higher multiple is used, while high confidence
  (high precision) means a lower multiple is used. Furthermore, we don't use
  any absolute distance threshold, because we assume that at least one icon
  is present in the image.

  Arguments:
      sc_distances: list of shape context distances.
      desired_confidence: The desired confidence for the bounding boxes that
         are returned, from 0 to 1. (default: {0.5})

  Returns:
      distance threshold - (float) threshold that can be used by clients to
        filter out distances that are above that threshold.
  """
  precision_interval_length = _OPTIMAL_ACCURACY_MULTIPLIER - _HIGH_PRECISION_MULTIPLIER
  recall_interval_length = _HIGH_RECALL_MULTIPLIER - _OPTIMAL_ACCURACY_MULTIPLIER

  # optimize for accuracy if confidence is not super high or low
  if desired_confidence == _MIDDLE_CONFIDENCE_LEVEL:
    relative_distance_multiplier = _OPTIMAL_ACCURACY_MULTIPLIER

  # optimize for recall if confidence is low
  elif desired_confidence < _MIDDLE_CONFIDENCE_LEVEL:
    # map desired confidence from [0, middle confidence level) to (0, 1]
    percent_recall_interval_length = desired_confidence / _MIDDLE_CONFIDENCE_LEVEL
    # start with the optimal accuracy multiplier and increase the multiplier
    # if desired confidence is low, up to the high-recall multiplier
    relative_distance_multiplier = _OPTIMAL_ACCURACY_MULTIPLIER + (
        percent_recall_interval_length * recall_interval_length)

  # optimize for precision if confidence is high
  elif desired_confidence > _MIDDLE_CONFIDENCE_LEVEL:
    # map desired confidence from (middle confidence level, 1] to [0, 1)
    percent_precision_interval_length = (1 - desired_confidence) / (
        1 - _MIDDLE_CONFIDENCE_LEVEL)
    # start with the high precision multiplier and increase the multiplier
    # if desired confidence is low, up to the optimal accuracy multiplier
    relative_distance_multiplier = _HIGH_PRECISION_MULTIPLIER + (
        percent_precision_interval_length * precision_interval_length)

  relative_max_dist = relative_distance_multiplier * min(sc_distances)
  return relative_max_dist


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


def _pad_bounding_box(icon_horizontal_padding_ratio: float,
                      icon_vertical_padding_ratio: float, image_height: int,
                      image_width: int, bbox: BoundingBox) -> BoundingBox:
  """Adds padding to bounding box according to padding ratios provided.

  Ensures that the padding added to the bounding box will not make it
  out of the image's height and width bounds. Takes the horizontal and
  vertical padding ratio and multiplies by the bounding box's current
  width and height to determine how much padding to add to the left/right
  and top/bottom, respectively.

  Arguments:
      icon_horizontal_padding_ratio: padding to width ratio for *both*
       the left and right side.
      icon_vertical_padding_ratio: padding to height ratio for *both*
      the top and the bottom.
      image_height: height of the image
      image_width: width of the image
      bbox: unpadded bounding box

  Returns:
      The original bounding box, except with dimensions altered to
      include padding as long as the image's height/width allow it.
  """
  bbox_width = bbox.max_x - bbox.min_x + 1
  bbox_height = bbox.max_y - bbox.min_y + 1
  bbox_horizontal_padding = int(bbox_width * icon_horizontal_padding_ratio)
  bbox_vertical_padding = int(bbox_height * icon_vertical_padding_ratio)
  padded_max_x = min(image_width - 1, bbox.max_x + bbox_horizontal_padding)
  padded_min_x = max(0, bbox.min_x - bbox_horizontal_padding)
  padded_max_y = min(image_height - 1, bbox.max_y + bbox_vertical_padding)
  padded_min_y = max(0, bbox.min_y - bbox_vertical_padding)
  return BoundingBox(padded_min_x, padded_min_y, padded_max_x, padded_max_y)


def standardize_bounding_boxes_padding(
    proposed_boxes_unpadded: List[BoundingBox], icon_box_unpadded: BoundingBox,
    icon: np.ndarray, image: np.ndarray) -> List[BoundingBox]:
  """Add padding onto proposed bounding boxes according to the original icon's.

  Use the original template icon's padding ratio to add a padding around the
  proposed bounding boxes, essentially making each of them larger. We assume
  that the input template icon is centered in its background.

  Arguments:
      proposed_boxes_unpadded: list of proposed BoundingBoxes, without padding
      icon_box_unpadded: the unpadded bounding box of the template icon
       found in its original background via contouring
      icon: the original template icon, from which the icon_box_unpadded
       is found through contouring
      image: the UI image

  Returns:
      List[BoundingBox]: List of the original Bounding Boxes, except with
      dimensions altered to include proportional padding according to the
      template icon's padding ratios.
  """
  icon_height, icon_width = icon.shape[:2]
  icon_box_height = icon_box_unpadded.get_height()
  icon_box_width = icon_box_unpadded.get_width()
  icon_vertical_padding = max(0, (icon_height - icon_box_height) /
                              2)  # for both top and bottom
  icon_vertical_padding_ratio = icon_vertical_padding / icon_box_height
  icon_horizontal_padding = max(0, (icon_width - icon_box_width) /
                                2)  # for both left and right
  icon_horizontal_padding_ratio = icon_horizontal_padding / icon_box_width

  image_height, image_width = image.shape[:2]
  return [
      _pad_bounding_box(icon_horizontal_padding_ratio,
                        icon_vertical_padding_ratio, image_height, image_width,
                        bbox) for bbox in proposed_boxes_unpadded
  ]
