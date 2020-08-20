"""Contains utility classes and modules used for analysis.

This contains:
- labeling the number of points in a cluster on the image
- plotting the number of points as a histogram
"""
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
from modules.bounding_box import BoundingBox
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
    plt.imshow(image)
    plt.imsave("%s-%d.png" % (output_path, index), image)


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
  fig = plt.figure()
  plt.hist(counts)
  plt.title(title)
  plt.xlabel("%s. Median: %f" % (xlabel, np.median(counts)))
  plt.ylabel("Frequency")
  plt.savefig(output_path)
  plt.close(fig=fig)


def save_icon_with_image(icon: np.ndarray, image: np.ndarray, filename: str):
  """Save icon and image side by side.

  Arguments:
      icon: numpy array representing the icon
      image: numpy array representing the UI image
      filename: filename for where to save the icon and image.
  """
  fig, ax = plt.subplots(figsize=(20, 10))
  ax.imshow(image)
  ax.axis("off")
  fig.figimage(icon, 0, 0)
  plt.savefig(filename, bbox_inches="tight", pad_inches=0.5)
  plt.close(fig=fig)


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
  fig = plt.figure()
  if connect_points:
    plt.plot(x, y, linestyle="solid")
  else:
    plt.scatter(x, y)
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.savefig(output_path)
  plt.close(fig=fig)


def scale_images_and_bboxes(
    images: List[np.ndarray], bboxes: List[List[BoundingBox]],
    horizontal_scale_factor: float, vertical_scale_factor: float
) -> Tuple[np.ndarray, List[List[BoundingBox]]]:
  """Scale a list of images and bounding boxes by scale factors given.

  Arguments:
      images: list of images to scale
      bboxes: list of bounding box lists to scale
      horizontal_scale_factor: horizontal scaling factor
      vertical_scale_factor: vertical scaling factor

  Returns:
      Tuple[scaled images, scaled bounding box lists scaled according to the
      horizontal and vertical scaling factors.
  """
  scaled_images = []
  scaled_bboxes = []
  for image, bbox_list in zip(images, bboxes):
    orig_height, orig_width = image.shape[:2]
    scaled_height = int(orig_height * vertical_scale_factor)
    scaled_width = int(orig_width * horizontal_scale_factor)
    # opencv sizes are width by height instead of height by width
    scaled_image = cv2.resize(src=image, dsize=(scaled_width, scaled_height))
    scaled_images.append(scaled_image)
    
    scaled_bbox_list = []
    for bbox in bbox_list:
      scaled_bbox_list.append(
          BoundingBox(bbox.min_x * horizontal_scale_factor,
                      bbox.min_y * vertical_scale_factor,
                      bbox.max_x * horizontal_scale_factor,
                      bbox.max_y * vertical_scale_factor))
    scaled_bboxes.append(scaled_bbox_list)
  return scaled_images, scaled_bboxes
