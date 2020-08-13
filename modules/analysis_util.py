"""Contains utility classes and modules used for analysis.

This contains:
- labeling the number of points in a cluster on the image
- plotting the number of points as a histogram
"""
from typing import List

import cv2
import matplotlib
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


def generate_histogram(samples: np.ndarray, title: str, xlabel: str,
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
  matplotlib.pyplot.hist(counts)
  matplotlib.pyplot.title(title)
  matplotlib.pyplot.xlabel("%s. Median: %f" % (xlabel, np.median(counts)))
  matplotlib.pyplot.ylabel("Frequency")
  matplotlib.pyplot.savefig(output_path)


def save_icon_with_image(icon: np.ndarray, image: np.ndarray, filename: str):
  """Save icon and image side by side.

  Arguments:
      icon: numpy array representing the icon
      image: numpy array representing the UI image
      filename: filename for where to save the icon and image.
  """
  fig, ax = matplotlib.pyplot.subplots(figsize=(20, 10))
  ax.imshow(image)
  ax.axis("off")
  fig.figimage(icon, 0, 0)
  matplotlib.pyplot.savefig(filename, bbox_inches="tight", pad_inches=0.5)
  matplotlib.pyplot.close(fig=fig)


def generate_scatterplot(x: np.ndarray,
                         y: np.ndarray,
                         title: str,
                         xlabel: str,
                         ylabel: str,
                         output_path: str,
                         connect_points: bool = True):
  """Utility to generate a scatterplot and save to file.

  Arguments:
      x: an array for x-values of each point
      y: an array for y-values of each point
      title: title of the scatterplot
      xlabel: title of x axis of scatterplot
      ylabel: title of y axis of scatterplot
      output_path: file path to save the scatter plot to
      connect_points: whether to conenct the points in the scatter
       plot (default: True)
  """
  fig = matplotlib.pyplot.figure()
  if connect_points:
    matplotlib.pyplot.plot(x, y, linestyle="solid")
  else:
    matplotlib.pyplot.scatter(x, y)
  matplotlib.pyplot.title(title)
  matplotlib.pyplot.xlabel(xlabel)
  matplotlib.pyplot.ylabel(ylabel)
  matplotlib.pyplot.savefig(output_path)
  matplotlib.pyplot.close(fig=fig)
