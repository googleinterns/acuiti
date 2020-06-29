import unittest


class TestStringMethods(unittest.TestCase):
  """This is a sample test class that tests string methods."""

  def test_upper(self):
    """This is a sample test function that tests string uppercase."""
    self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
  unittest.main()
