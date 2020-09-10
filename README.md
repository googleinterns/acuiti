**NOTE: This is not an officially supported Google product.**

# Icon Matching with Shape Context Descriptors

This project involves using shape-context descriptors to find an arbitrary template icon in a given image. The overall algorithm has three steps: 1) edge detection, 2) clustering contours, and 3) using shape context descriptors to find the closest contour to the template icon. Shape context descriptors were introduced in this 2001 research paper by S. Belongie et al: https://papers.nips.cc/paper/1913-shape-context-a-new-descriptor-for-shape-matching-and-object-recognition.pdf and its implementation is available in OpenCV: https://docs.opencv.org/master/d8/de3/classcv_1_1ShapeContextDistanceExtractor.html. Here's an overview of the process:

<p align="center">
  <img src="https://github.com/googleinterns/acuiti/blob/update-README/docs/Algorithm-Overview.png"/></p>

<p align="center">Algorithm Overview<p align="center">


This project built and optimized the algorithm pipeline described above to achieve a find icon algorithm that is:
- [x] scale-invariant
- [x] color-invariant
- [x] has ~95% recall/precision 
- [x] takes about 1.5-2.5s on average

# Repository Overview
Here's an overview of the files in this repository. There are three main types of files of interest (in bold) which are explained in further detail in each of the sections below.

Under ```modules/```:
  - **Benchmark Pipeline**, which runs any find icon algorithm on any dataset and outputs accuracy, latency, and memory information;
  - **Find Icon Algorithms**, which includes our optimized implementation of the shape context descriptor algorithm that achieves ~95% recall/precision in 1-2s on average;
  - **Analysis Utilities**, which are tools used to run experiments to figure out how to optimize our shape context descriptor algorithm;

Under ```tests/```:
- Integration and Unit Tests, which test the functionalities above.

Under ```datasets/```:
- Small datasets used for integration tests. Actual datasets to validate results are much larger and not included in this repository.

# Benchmark Pipeline
The end-to-end pipeline can be run from the command-line ```python -m modules.benchmark_pipeline``` with the following flags:

```
usage: benchmark_pipeline.py [-h] [--tfrecord_path TFRECORD_PATH] [--iou_threshold THRESHOLD] [--output_path OUTPUT_PATH] [--multi_instance_icon MULTI_INSTANCE_ICON] [--visualize VISUALIZE]

Run a benchmark test on find_icon algorithm.

optional arguments:
  -h, --help            show this help message and exit
  --tfrecord_path TFRECORD_PATH
                        path to tfrecord (default: datasets/small_single_instance_v2.tfrecord)
  --iou_threshold THRESHOLD
                        iou above this threshold is considered accurate (default: 0.600000)
  --output_path OUTPUT_PATH
                        path to where output is written (default: )
  --multi_instance_icon MULTI_INSTANCE_ICON
                        whether to evaluate with multiple instances of an icon in an image (default: False)
  --visualize VISUALIZE
                        whether to visualize bounding boxes on image (default: False)
 ```
 
The benchmark pipeline can be modified with these files:
- ```modules/benchmark_pipeline.py``` which has the end-to-end pipeline, including a visualization option
- ```modules/util.py``` which has tools to read in a dataset from a TfRecord file and custom Latency and Memory-tracking classes
- ```modules/defaults.py``` can be modified to change the default icon finder algorithm, IOU threshold, output path, and dataset path to run the benchmark pipeline with.

# Find Icon Algorithms
A custom find icon algorithm can be passed into the benchmark pipeline when run programmatically. These are the relevant files:
- ```modules/algorithms.py``` includes a suite of algorithms for edge detection, shape context descriptor distance calculation, precision & recall calculation
- ```modules/icon_finder.py``` is the abstract base class that the custom find icon algorithm should inherit from. 
- ```modules/icon_finder_shape_context.py``` is the optimized version of the shape context algorithm pipeline that we used to achieve our current metrics
- ```modules/clustering_algorithms.py``` contains wrappers for Sklearn's clustering algorithms with custom defaults exposed for our use cases

# Analysis Utilities
Analysis tools are provided in the following files:
- ```modules/analysis_util.py``` contains tools to label cluster sizes, generate histograms, saving an icon/image as a pair, generate scatterplots, and scaling images/bounding boxes
- ```modules/optimizer.py``` contains an optimizer to find best hyperparameters for clustering algorithms
