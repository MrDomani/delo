import unittest
from delo import DescribedFunction
import numpy as np
from delo import CustomExceptions


def restrigin5D(x):
    dim = 5
    if len(x.shape) == 1:
        raise IOError(f"x is 1D vector, but should be a 2D matrix: x = {x}")
    if x.shape[1] != dim:  # x is np.array with dim columns
        raise IOError(f"there should be {dim} columns of x, but: x = {x}")
    return 10 * dim + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x), axis=1)

class TestDescribedFunction(unittest.TestCase):
    def test_functionNotCollable(self):
        with self.assertRaises(CustomExceptions.FunctionNoCollableException):
            DescribedFunction(10, 10)

    def test_improperDimention(self):
        with self.assertRaisesRegex(CustomExceptions.VariableNotIntException, "str"):
            DescribedFunction(restrigin5D, "5")
        with self.assertRaisesRegex(CustomExceptions.VariableNotIntException, "float"):
            DescribedFunction(restrigin5D, 0.2)
        with self.assertRaisesRegex(CustomExceptions.VariableNotIntException, "float"):
            DescribedFunction(restrigin5D, 1.0)
        with self.assertRaisesRegex(CustomExceptions.ImproperIntException, "0"):
            DescribedFunction(restrigin5D, 0)
        with self.assertRaisesRegex(CustomExceptions.ImproperIntException, "-1"):
            DescribedFunction(restrigin5D, -1)

        with self.assertRaises(CustomExceptions.ImproperDomainLimitsException):
            DescribedFunction(restrigin5D, 5, domain_lower_limit=4, domain_upper_limit=3)
        with self.assertRaisesRegex(CustomExceptions.ImproperDomainLimitsException, "equal lower and upper limits"):
            DescribedFunction(restrigin5D, 5,
                              domain_lower_limit=np.array((3,3,3,3,3)),
                              domain_upper_limit=np.array((3,4,4,4,4)))

if __name__ == '__main__':
    unittest.main()