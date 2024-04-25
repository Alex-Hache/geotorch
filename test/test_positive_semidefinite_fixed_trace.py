from unittest import TestCase

from geotorch.pssdfixedranktrace import PSSDFixedRankTrace


class TestPSSDFixedTrace(TestCase):
    def test_positive_semidefinite_fixed_trace_errors(self):
        cls = PSSDFixedRankTrace
        # rank always has to be 1 <= rank <= n
        with self.assertRaises(ValueError):
            cls(size=(4, 4), rank=5)
        with self.assertRaises(ValueError):
            cls(size=(3, 3), rank=0)
        # Instantiate it in a non-square matrix
        with self.assertRaises(ValueError):
            cls(size=(3, 6), rank=2)
        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            cls(size=(5,), rank=1)

        # Try to instantiate it in a vector rather than a matrix
        with self.assertRaises(ValueError):
            cls(size=(5,), rank=2)
        # Or a non-square
        with self.assertRaises(ValueError):
            cls(size=(5, 3), rank=2)

        # Pass a non-callable object
        with self.assertRaises(ValueError):
            cls(size=(5, 2), rank=1, f=3)
        # Or the wrong string
        with self.assertRaises(ValueError):
            cls(size=(5, 3), rank=2, f="fail")
        # Same with PSD
        with self.assertRaises(ValueError):
            cls(size=(5, 2), rank=2, f=3)
        with self.assertRaises(ValueError):
            cls(size=(5, 3), rank=2, f="fail")
