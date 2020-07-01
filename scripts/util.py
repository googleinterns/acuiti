"""Contains utility classes.

So far, we have classes that measure latency and
memory usage.
"""
import sys
import cProfile
import io
import pstats
from pstats import SortKey
from memory_profiler import memory_usage


class LatencyTimer:
  """Wrapper class for cython runtime profiler.
  """

  def __init__(self):
    self.pr = cProfile.Profile()

  def start(self):
    self.pr.enable()

  def stop(self):
    self.pr.disable()

  def print_info(self, output_file=sys.stdout):
    """Prints memory usage information.

    Keyword Arguments:
        output_file {file object} --
         file object to print output to. (default: {sys.stdout})
    """
    s = io.StringIO()
    ps = pstats.Stats(self.pr, stream=s).sort_stats(SortKey.CUMULATIVE)
    ps.print_stats()
    output_file.write(s.getvalue())
    output_file.flush()


class MemoryTracker:
  """Wrapper class for PyPI's memory-profiler.
  """

  def __init__(self):
    self.memory_info = ""

  def run_and_track_memory(self, func_args_tuple):
    """Tracks memory usage of a function.

    Arguments:
        func_args_tuple {tuple} -- tuple of function and
         variable number of arguments to the function
    """
    self.memory_info = memory_usage(func_args_tuple)

  def print_info(self, output_file=sys.stdout):
    """Prints memory usage information.

    Keyword Arguments:
        output_file {file object} --
         file object to print output to. (default: {sys.stdout})
    """
    output_file.write(self.memory_info)
    output_file.flush()
