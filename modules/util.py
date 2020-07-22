"""Contains utility classes.

So far, we have classes that measure latency and
memory usage.
"""
import cProfile
import io
import pstats

import memory_profiler

import modules.defaults as defaults
import numpy as np


class LatencyTimer:
  """Wrapper class for cython runtime profiler.
  """

  def __init__(self):
    self.pr = cProfile.Profile()

  def start(self):
    self.pr.enable()

  def stop(self):
    self.pr.disable()

  def calculate_info(self, output_path: str = defaults.OUTPUT_PATH) -> float:
    """Calculates latency info and optionally prints to file.

    Args:
        output_path: file path to print output to.
         (default: None)

    Returns:
        float -- the total seconds taken between calls to start and stop
    """
    s = io.StringIO()
    sort_by = pstats.SortKey.CUMULATIVE  # pytype: disable=module-attr
    ps = pstats.Stats(self.pr, stream=s).sort_stats(sort_by)
    ps.print_stats()
    info = s.getvalue()
    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write(info)
    # parse cProfiler's output to get the total time as a float
    first_line = info.partition("\n")[0]
    total_time = first_line.split(" ")[-2]
    return float(total_time)


class MemoryTracker:
  """Wrapper class for PyPI's memory-profiler.
  """

  def __init__(self):
    self.memory_info = ""

  def run_and_track_memory(self, func_args_tuple):
    """Tracks memory usage of a function.

    Args:
        func_args_tuple: tuple of function and
         variable number of arguments to the function
          in the form of (f, args, kw)
          Example argument: (f, (1,), {'n': int(1e6)})
    """
    self.memory_info = memory_profiler.memory_usage(func_args_tuple)

  def calculate_info(self, output_path: str = defaults.OUTPUT_PATH) -> float:
    """Calculates memory usage info and optionally prints to file.

    Args:
        output_path: file path to print output to.
         (default: None)

    Returns:
        float -- the MiBs used by the function call
    """
    average_mb = self.memory_info
    if len(self.memory_info) > 1:
      average_mb = np.mean(self.memory_info)
    output_msg = "Process took %f MiBs \n" % average_mb
    if output_path:
      with open(output_path, "a") as output_file:
        output_file.write(output_msg)
    return average_mb
