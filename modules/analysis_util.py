"""Contains utility classes and modules used for analysis.

This contains:
- labeling the number of points in a cluster on the image
- plotting the number of points as a histogram
"""
from typing import List

import cv2
import matplotlib.pyplot
import numpy as np


def label_cluster_size(image_clusters: List[np.ndarray],
                       image_list: List[np.ndarray], output_path: str):
  """For each image, label all its clusters with cluster sizes.

  Arguments:
      image_clusters: List of list of clusters corresponding to each image
      image_list: List of images
      output_path: File path where each image is stored
  """
  font = cv2.FONT_HERSHEY_PLAIN
  color = (255, 0, 0)  # blue
  thickness = 2  # number of pixels
  font_scale = 1  # font multiplier
  # for each image, label each of its clusters with its size
  for index, (clusters, image) in enumerate(zip(image_clusters, image_list)):
    for contour in clusters:
      contours_poly = cv2.approxPolyDP(contour, 3, True)
      x, y, _, _ = cv2.boundingRect(contours_poly)
      bottom_left = (x, y)
      text = str(len(contour))
      cv2.putText(image, text, bottom_left, font, font_scale, color, thickness)
    matplotlib.pyplot.imshow(image)
    matplotlib.pyplot.imsave("%s-%d.png" % (output_path, index), image)


def generate_histogram(samples: np.ndarray,
                       title: str,
                       xlabel: str,
                       output_path: str):
  """Generate a histogram based off of samples.

  Arguments:
      samples: a potentially multi-dimensional array of samples. This is treated
      as a one-dimensional array, where each entry is a sample whose value is
      bucketed when creating the histogram.
      title: title for the histogram
      xlabel: label for the x_axis of the histogram (the y_axis is frequency)
      output_path: file path for resulting histogram plot to be saved at
  """
  counts = samples.flatten()
  fig = matplotlib.pyplot.figure()
  matplotlib.pyplot.hist(counts)
  matplotlib.pyplot.title(title)
  matplotlib.pyplot.xlabel("%s. Median: %f" % (xlabel, np.median(counts)))
  matplotlib.pyplot.ylabel("Frequency")
  matplotlib.pyplot.savefig(output_path)
  matplotlib.pyplot.close(fig=fig)
