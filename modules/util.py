"""Contains utility classes.

So far, we have classes that measure latency and
memory usage.
"""
import cProfile
import io
import pstats

from memory_profiler import memory_usage

from modules.defaults import DEFAULT_ARGS
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

  def print_info(self, output_path=DEFAULT_ARGS["output_path"]):
    """Prints memory usage information.

    Args:
        output_path: file path to print output to.
         (default: {out.txt})

    Returns:
        float -- the total time taken between calls to start and stop
    """
    s = io.StringIO()
    sort_by = pstats.SortKey.CUMULATIVE  # pytype: disable=module-attr
    ps = pstats.Stats(self.pr, stream=s).sort_stats(sort_by)
    ps.print_stats()
    info = s.getvalue()
    output_file = open(output_path, "a")
    output_file.write(info)
    output_file.close()
    print(info)
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
    """
    self.memory_info = memory_usage(func_args_tuple)

  def print_info(self, output_path=DEFAULT_ARGS["output_path"]):
    """Prints memory usage information.

    Args:
        output_path: file path to print output to.
         (default: {out.txt})

    Returns:
        float -- the megabytes used by the function call
    """
    average_mb = self.memory_info
    if len(self.memory_info) > 1:
      average_mb = np.mean(self.memory_info)
    output_file = open(output_path, "a")
    output_msg = "Process took %f megabytes \n" % average_mb
    output_file.write(output_msg)
    output_file.close()
    print(output_msg)
    return average_mb
