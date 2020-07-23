"""This file contains utility functions for find icon computer vision algorithms."""
from typing import List, Tuple

import cv2
from modules.bounding_box import BoundingBox
import numpy as np
import sklearn.cluster


def shape_context_descriptor(icon_contour: np.ndarray,
                             img_contour: np.ndarray) -> float:
  """Calculates the shape context distance bewteen two contours.

  Arguments:
      icon_contour: A list with shape (n, 1, 2).
       Represents the template icon contour
      img_contour: A list with shape (n, 1, 2).
       Represents the image patch contour.
       (Note: function will fail unless the number of channels is 2.)

  Returns:
      float: the shape context distance between the two contours.
  """
  extractor = cv2.createShapeContextDistanceExtractor()
  return extractor.computeDistance(icon_contour, img_contour)


def detect_contours(image: np.ndarray,
                    use_bilateral_filter: bool) -> List[List[List[int]]]:
  """Detects the contours in the image.

  Arguments:
      image: Input image, as ndarray.
      use_bilateral_filter: whether to use bilateral filter
       to smooth out edges before Canny edge detection.

  Returns:
      List of contour groups,each of which is a list of points.
  """
  imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  if use_bilateral_filter:
    imgray = cv2.bilateralFilter(imgray, 5, 20, 50)
  edges = cv2.Canny(imgray, 10, 25)
  img_contours, _ = cv2.findContours(edges, cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_NONE)
  return img_contours


def cluster_contours_dbscan(
    img_contours: np.ndarray,
    eps: float = 10,
    min_samples: int = 5) -> Tuple[List[np.ndarray], List[np.ndarray]]:
  """Group contours using DBSCAN.

  Arguments:
      img_contours: Flattened list of all points in all contours of an image.
      That is: List[List[int]]
      eps: The maximum distance a point can be away to be considered
       within neighborhood of another point. (default: {10})
      min_samples: The number of points needed within a neighborhood
       of a point for it to be a core point. (default: {5})

  Returns:
      List of groups of points, each representing its own contour,
       and a masked list of groups of bool values, each representing
        which points are core points or not. The contours and masks
        are np.ndarrays of roughly List[List[int]] and List[bool].
  """
  clusters = sklearn.cluster.DBSCAN(eps=eps,
                                    min_samples=min_samples).fit(img_contours)
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
    contour_group = img_contours[np.argwhere(clusters.labels_ == i)]
    contour_groups.append(np.vstack(contour_group).squeeze())
    core_samples_mask_groups.append(
        np.vstack(
            core_samples_mask[np.argwhere(clusters.labels_ == i)]).squeeze())
  return contour_groups, core_samples_mask_groups


def get_nms_bounding_boxes(
    contours: np.ndarray,
    confidences: np.ndarray,
    confidence_threshold: float,
    nms_threshold: float = 0.9,
) -> List[BoundingBox]:
  """Returns bounding boxes after filtering through non-max suppression.

  Arguments:
      contours: list of contours (which is a list of x,y points) to filter.
      confidences: confidence scores associated with each contour.
      confidence_threshold: only keep bboxes with
       confidence scores (strictly) above this threshold.
      nms_threshold: two bboxes with an IOU >=
       this threshold are considered the same bbox. (default: 0.9)

  Returns:
      List of BoundingBoxes that passed the filter.
  """
  bboxes = []
  rects = []
  for contour in contours:
    contours_poly = cv2.approxPolyDP(contour, 3, True)
    bound_rect = cv2.boundingRect(contours_poly)
    rects.append(bound_rect)
    bbox = BoundingBox(bound_rect[0], bound_rect[1],
                       bound_rect[0] + bound_rect[2],
                       bound_rect[1] + bound_rect[3])
    bboxes.append(bbox)
  indices = cv2.dnn.NMSBoxes(rects, confidences, confidence_threshold,
                             nms_threshold)
  return [bboxes[i[0]] for i in indices]
