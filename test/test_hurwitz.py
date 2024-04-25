from unittest import TestCase

from geotorch.hurwitz import Hurwitz


class TestHurwitz(TestCase):
    def test_hurwitz_errors(self):
        cls = Hurwitz
        # alpha always has to be strictly positive
        with self.assertRaises(ValueError):
            cls(size=(4, 4), alpha=-2)
        with self.assertRaises(ValueError):
            cls(size=(3, 3), alpha=0)
        # Instantiate it in a non-square matrix
        with self.assertRaises(ValueError):
            cls(size=(3, 6), alpha=2)
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            cls(size=(5,), alpha=1)
