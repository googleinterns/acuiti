**NOTE: This is not an officially supported Google product.**

# Icon Matching with Shape Context Descriptors

This project involves using shape-context descriptors to find an arbitrary template icon in a given image. The overall algorithm has three steps: 1) edge detection, 2) clustering contours, and 3) using shape context descriptors to find the closest contour to the template icon. Shape context descriptors were introduced in this 2001 research paper by S. Belongie et al: https://papers.nips.cc/paper/1913-shape-context-a-new-descriptor-for-shape-matching-and-object-recognition.pdf and its implementation is available in OpenCV: https://docs.opencv.org/master/d8/de3/classcv_1_1ShapeContextDistanceExtractor.html. Here's an overview of the process:

<p align="center">
  <img src="https://github.com/googleinterns/acuiti/blob/update-README/docs/Algorithm-Overview.png"/></p>

<p align="center">Algorithm Overview<p align="center">


This project built and optimized the algorithm pipeline described above to achieve an icon matching algorithm that is:
- [x] scale-invariant
- [x] color-invariant
- [x] has ~95% recall/precision 
- [x] takes about 1.5-2.5s on average

# Repository Overview
Here's an overview of the files in this repository. There are three main types of files of interest (in bold) which are explained in further detail in each of the sections below.

Under ```modules/```:
  - **Benchmark Pipeline**, which runs any icon matching algorithm on any dataset and outputs accuracy, latency, and memory information;
  - **Icon Matching Algorithms**, which includes our optimized implementation of the shape context descriptor algorithm that achieves ~95% recall/precision in 1-2s on average;
  - **Analysis Utilities**, which are tools used to run experiments to figure out how to optimize our shape context descriptor algorithm;

Under ```tests/```:
- Integration and Unit Tests, which test the functionalities above.

Under ```datasets/```:
- Small datasets used for integration tests. Actual datasets to validate results are much larger and not included in this repository.

# Benchmark Pipeline
## Running from the Command-Line
The end-to-end pipeline can be run from the command-line as such:
```python -m modules.benchmark_pipeline --tfrecord_path=datasets/small_single_instance_v2.tfrecord --output_path=small_single_instance.txt --multi_instance_icon=False --visualize=True --iou_threshold=0.6```. 

The results (accuracy, precision, recall, latency average/median, memory average/median) will then be printed to the output txt file as well as to stdout like so:
```
Average seconds per image: 1.439400
Median seconds of images: 1.544500

Average MiBs per image: 6.865234
Median MiBs per image: 5.380859

Accuracy: 0.935484

Precision: 0.966667

Recall: 0.966667
```
The output txt file will additionally contain latency profiling information for the icon matching algorithm. The memory calculated is the *auxiliary memory* needed by the icon matching algorithm.

Here are more details on the flags:

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
 
 ## Running Programmatically
When run programmatically, the benchmark pipeline can also support some additional parameters, such as a custom icon detection algorithm. Here's an example:
```
benchmark = BenchmarkPipeline(tfrecord_path="datasets/small_multi_instance_v2.tfrecord")
correctness, latency_avg_secs, memory_avg_mibs = benchmark.evaluate(icon_finder_object=
                                                                icon_finder_shape_context.IconFinderShapeContext(clusterer=clustering_algorithms.DBSCANClusterer()))
```
(Note that correctness is a dataclass from which we can extract accuracy, precision, and recall by calling ```correctness.accuracy```, ```correctness.precision```, ```correctness.recall```). Example usage of the benchmark pipeline for multi-instance cases can also be found in ```tests/integration_tests.py```. 

## Modifying the Pipeline
The benchmark pipeline can be modified with these files:
- ```modules/benchmark_pipeline.py``` which has the end-to-end pipeline, including a visualization option
- ```modules/util.py``` which has tools to read in a dataset from a TfRecord file and custom Latency and Memory-tracking classes
- ```modules/defaults.py``` can be modified to change the default icon finder algorithm, IOU threshold, output path, and dataset path to run the benchmark pipeline with.

# Icon Matching Algorithms
A custom icon matching algorithm can be passed into the benchmark pipeline when run programmatically. These are the relevant files:
- ```modules/algorithms.py``` includes a suite of algorithms for edge detection, shape context descriptor distance calculation, precision & recall calculation
- ```modules/icon_finder.py``` is the abstract base class that the custom icon matching algorithm should inherit from. 
- ```modules/icon_finder_shape_context.py``` is the optimized version of the shape context algorithm pipeline that we used to achieve our current metrics (and can be run as a standalone).
- ```modules/clustering_algorithms.py``` contains wrappers for Sklearn's clustering algorithms with custom defaults exposed for our use cases

# Analysis Utilities
Analysis tools are provided in the following files:
- ```modules/analysis_util.py``` contains tools to label cluster sizes, generate histograms, saving an icon/image as a pair, generate scatterplots, and scaling images/bounding boxes
- ```modules/optimizer.py``` contains an optimizer to find best hyperparameters for clustering algorithms
