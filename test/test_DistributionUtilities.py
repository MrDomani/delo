import unittest
from delo import DistributionUtilities
import numpy as np
import inspect
import unittest

def getVerbosity():
    """Return the verbosity setting of the currently running unittest
    program, or 0 if none is running.

    when `python -m unittest` is called, this function should return 1. when `python -m unittest -v`, then 2
    """
    frame = inspect.currentframe()
    while frame:
        self = frame.f_locals.get('self')
        if isinstance(self, unittest.TestProgram):
            return self.verbosity
        frame = frame.f_back
    return 0


class TestChoppedNormal(unittest.TestCase):
    def setUp(self):
        rng_for_seed = np.random.default_rng()  # should work for any seed
        seed = rng_for_seed.integers(1, 100000)
        if getVerbosity() > 1:
            print(f"Test ChoppedNormal on seed = {seed}")
        self.seed = seed
        self.size = 1000

    def test_in_bounds(self):
        rng = np.random.default_rng(self.seed)
        drawned = DistributionUtilities.chopped_normal(rng, self.size, np.ones(self.size))
        self.assertTrue(np.all(drawned <= 1))
        self.assertTrue(np.all(drawned >= 0))

        drawned = DistributionUtilities.chopped_normal(rng, self.size, np.zeros(self.size))
        self.assertTrue(np.all(drawned <= 1))
        self.assertTrue(np.all(drawned >= 0))

    def test_reproducibility(self):
        rng1 = np.random.default_rng(self.seed)
        rng2 = np.random.default_rng(self.seed)

        drawned1 = DistributionUtilities.chopped_normal(rng1, self.size)
        drawned2 = DistributionUtilities.chopped_normal(rng2, self.size)

        self.assertTrue(np.all(drawned1 == drawned2))

    def test_zero_variation(self):
        rng = np.random.default_rng(self.seed)
        np.testing.assert_array_equal(DistributionUtilities.chopped_normal(rng, self.size, variation=0),
                                      0.5 * np.ones(self.size))



class TestChoppedCauchy(unittest.TestCase):
    def setUp(self):
        rng_for_seed = np.random.default_rng()  # should work for any seed
        seed = rng_for_seed.integers(1, 100000)
        if getVerbosity() > 1:
            print(f"Test ChoppedCauchy on seed = {seed}")
        self.seed = seed
        self.size = 1000

    def test_in_bounds(self):
        rng = np.random.default_rng(self.seed)

        drawned = DistributionUtilities.chopped_cauchy(rng, self.size, np.ones(self.size))
        self.assertTrue(np.all(drawned <= 1))
        self.assertTrue(np.all(drawned >= 0))

        drawned = DistributionUtilities.chopped_cauchy(rng, self.size, np.zeros(self.size))
        self.assertTrue(np.all(drawned <= 1))
        self.assertTrue(np.all(drawned >= 0))

    def test_reproducibility(self):
        rng1 = np.random.default_rng(self.seed)
        rng2 = np.random.default_rng(self.seed)

        drawned1 = DistributionUtilities.chopped_cauchy(rng1, self.size)
        drawned2 = DistributionUtilities.chopped_cauchy(rng2, self.size)

        self.assertTrue(np.all(drawned1 == drawned2))

    def test_zero_scale(self):
        rng = np.random.default_rng(self.seed)
        np.testing.assert_array_equal(DistributionUtilities.chopped_cauchy(rng, self.size, scale=0),
                                      0.5 * np.ones(self.size))

class TestChooseIndicesForMutationDraw(unittest.TestCase):
    def setUp(self):
        rng_for_seed = np.random.default_rng()  # should work for any seed
        seed = rng_for_seed.integers(1, 100000)
        if getVerbosity() > 1:
            print(f"Test ChooseIndicesForMutationDraw on seed = {seed}")
        self.seed = seed

    def test_reproducibility(self):
        rng = np.random.default_rng(self.seed)
        population_size = rng.integers(1, 1000)
        archive_size = rng.integers(0, 100)

        rng1 = np.random.default_rng(self.seed)
        rng2 = np.random.default_rng(self.seed)

        indices_1_1, indices_2_1 = DistributionUtilities.choose_2_columns_of_integers(rng1, nrow=population_size,
                                                                                      matrix_of_restrictions=np.array(
            [[0, 0], [population_size, population_size+archive_size]]))
        indices_1_2, indices_2_2 = DistributionUtilities.choose_2_columns_of_integers(rng2, nrow=population_size,
                                                                                      matrix_of_restrictions=np.array(
            [[0, 0], [population_size, population_size+archive_size]]))

        self.assertTrue(np.all(indices_1_1 == indices_1_2))
        self.assertTrue(np.all(indices_2_1 == indices_2_2))

class TestEditedQuantile(unittest.TestCase):
    def test_Edited_Quantile_full(self):
        my_array = np.array([0,1,2,3,4,5,6,7,8,9])
        self.assertEqual(DistributionUtilities.edited_quantile(my_array, 0.5),
                         4.5)
        self.assertEqual(DistributionUtilities.edited_quantile(my_array, 0.1),
                         0.9)
        self.assertEqual(DistributionUtilities.edited_quantile(my_array, 0.0),
                         0.0)
        self.assertEqual(DistributionUtilities.edited_quantile(my_array, 1.0),
                         9.0)

    def test_Edited_Quantile_empty(self):
        empty_array = np.array([])
        self.assertEqual(DistributionUtilities.edited_quantile(empty_array, 1.0),
                         0)
        self.assertEqual(DistributionUtilities.edited_quantile(empty_array, 0.5),
                         0)
        self.assertEqual(DistributionUtilities.edited_quantile(empty_array, 0.0),
                         0)

if __name__ == '__main__':
    unittest.main()