import unittest
from delo import DElo, DescribedFunction, CustomExceptions
import numpy as np


def restrigin5D(x):
    dim = 5
    if len(x.shape) == 1:
        raise IOError(f"x is 1D vector, but should be a 2D matrix: x = {x}")
    if x.shape[1] != dim:  # x is np.array with dim columns
        raise IOError(f"there should be {dim} columns of x, but: x = {x}")
    return 10 * dim + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x), axis=1)


class TestAbstractDESetUp(unittest.TestCase):
    def setUp(self):
        rng_for_seed = np.random.default_rng()  # should work for any seed
        seed = rng_for_seed.integers(1, 100000)
        self.seed = seed
        self.solver = DElo(100)
        self.rng = np.random.default_rng(self.seed)
        self.restrigin5D = restrigin5D
        self.described_function = DescribedFunction(restrigin5D,
                                                    dimension=5,
                                                    domain_lower_limit=-5,
                                                    domain_upper_limit=5)

    def set_get_example_population_trial(self):
        self.solver.optimize(self.described_function, max_f_evals=-1, rng_seed=self.seed)
        U = self.rng.random((self.solver.population_size, self.solver.function.dimension))
        population_trial_org = self.solver.function.domain_lower_limit + U * (  # inside [-5, 5] boundary
                self.solver.function.domain_upper_limit - self.solver.function.domain_lower_limit)
        self.solver._population_trial = population_trial_org
        return population_trial_org


class TestAbstractDEInit(TestAbstractDESetUp):

    def test_check_init_correctness(self):
        with self.assertRaises(CustomExceptions.ImproperRestartEpsilonException):
            DElo(100, restart_eps_x=-5)
        with self.assertRaises(CustomExceptions.ImproperRestartEpsilonException):
            DElo(100, restart_eps_y=-5)
        with self.assertRaises(CustomExceptions.NonIntArchiveException):
            DElo(100, archive_size="11")
        with self.assertRaises(CustomExceptions.NonIntArchiveException):
            DElo(100, archive_size=4.5)
        with self.assertRaises(CustomExceptions.NegativeArchiveException):
            DElo(100, archive_size=-3)
        with self.assertRaises(CustomExceptions.PopulationSizeNotIntException):
            DElo("100")
        with self.assertRaises(CustomExceptions.NonPositivePopulationSizeException):
            DElo(0)
        with self.assertRaises(CustomExceptions.PopulationSizeLessEq4Exception):
            DElo(1)
        with self.assertRaises(CustomExceptions.p_best_rateOutOf01Exception):
            DElo(100, p_best_rate=-0.01)
        with self.assertRaises(CustomExceptions.p_best_rateOutOf01Exception):
            DElo(100, p_best_rate=1.1)

    def test_init_population(self):
        self.solver.function = self.described_function
        self.solver.rng = np.random.default_rng(self.seed)
        self.solver._init_population()
        self.assertIsInstance(self.solver._population, np.ndarray)
        self.assertTupleEqual(self.solver._population.shape,
                              (self.solver.population_size, self.solver.function.dimension))
        np.testing.assert_array_less(self.solver._population, np.tile(self.solver.function.domain_upper_limit + 1e-14,
                                                                      (self.solver.population_size, 1)))
        np.testing.assert_array_less(
            np.tile(self.solver.function.domain_lower_limit - 1e-14, (self.solver.population_size, 1)),
            self.solver._population)
        # Test randomness (kinda)
        self.assertGreater(np.unique(self.solver._population).size, self.solver._population.size * 0.7)

    def test_set_p_best(self):
        self.solver.population_size = 100
        self.solver._population_f_value = np.linspace(0, 1, num=self.solver.population_size, endpoint=False)
        self.solver.p_best_rate = 0.1
        self.solver._set_p_best()
        self.assertEqual(self.solver.current_worst_i, 99)
        self.assertEqual(self.solver.current_worst_f, 0.99)
        self.assertEqual(self.solver.current_best_i, 0)
        self.assertEqual(self.solver.current_best_f, 0.0)
        np.testing.assert_array_equal(self.solver.current_p_best_i, np.arange(10))
        np.testing.assert_array_equal(self.solver.current_p_best_f, np.linspace(0.0, 0.1, num=10, endpoint=False))


class TestAbstractDERestarts(TestAbstractDESetUp):

    def test_after_optimization(self):
        self.solver.optimize(self.described_function,
                             max_f_evals=-1, rng_seed=self.seed)  # should initialize, but not perform any generation

        self.solver.optimize(self.described_function, max_f_evals=1003, rng_seed=self.seed)
        np.testing.assert_array_equal(self.restrigin5D(self.solver._population),
                                      self.solver._population_f_value)

        best_member_ever1, best_f_value_ever1 = self.solver.get_solution()
        np.testing.assert_array_equal(best_f_value_ever1, np.min(self.solver._population_f_value))
        np.testing.assert_array_equal(best_member_ever1,
                                      self.solver._population[np.argmin(self.solver._population_f_value)])

        self.assertTrue(self.solver.restarts == 0)  # 1000 is too litle to cause the restart (almost surely)
        self.assertTrue(
            self.solver._generations_processed == 10 - 1)  # There were 1003 max_f_evals and 100 of them were used every generation + 100 was used during initization
        self.assertTrue(
            self.solver._remaining_evals == 3)  # After 10 generations, 3 was left and it was less than 100, so optimization stopped
        self.assertTrue(self.solver.archive.shape[
                            0] == self.solver.max_archive_size)  # after 3 +- 1 generations archive should be full
        self.assertTrue(self.solver.current_worst_f >= self.solver.current_best_f)  # the best is better than the worst

        self.solver.optimize(self.described_function, max_f_evals=10000, rng_seed=self.seed)  # 10 times the previous
        best_member_ever2, best_f_value_ever2 = self.solver.get_solution()
        self.assertTrue(
            best_f_value_ever2 < best_f_value_ever1)  # after bigger optimization, the found solution should be better

    def test_restarts(self):
        self.solver.optimize(self.described_function, max_f_evals=1000, rng_seed=self.seed)
        best_member_ever, best_f_value_ever = self.solver.get_solution()

        self.solver._restart_search()
        self.assertTrue(self.solver.restarts == 1)  # there was one non-initial restart and 1 initial one
        np.testing.assert_array_equal(self.solver.archive.shape, np.array([0,
                                                                            5]))  # archive is ampty; WARNING(once this test failed, but I didn't save the seed. Do this if it happen for future investigation)
        self.assertTrue(
            best_f_value_ever < self.solver.current_best_f)  # It is highly unlikely, the better solution will be drawn from random


class TestAbstractDESetUpTrim(TestAbstractDESetUp):
    def setUp(self):
        super().setUp()
        self.solver._population = self.set_get_example_population_trial()
        self.solver._population_trial = self.set_get_example_population_trial() * 1.5
        population_trial_org = self.solver._population_trial

    def test_does_not_trim_when_not_necessary(self):
        population_trial_org = self.set_get_example_population_trial()
        self.solver._trim_population_trial_to_domain()
        np.testing.assert_array_equal(self.solver._population_trial, population_trial_org)

    def test_trims_in_specific(self):
        population_trial_org = self.set_get_example_population_trial()
        population_trial = -10 * np.ones((self.solver.population_size,
                                          self.solver.function.dimension))  # outside of [-5, 5] boundary
        population_trial[0:(self.solver.population_size // 2), :] = 0
        self.solver._population = np.zeros(population_trial_org.shape)
        self.solver._population_trial = population_trial.copy()
        self.solver._trim_population_trial_to_domain()

        expected_population_trial = np.where(population_trial == -10, -5, 0)
        np.testing.assert_array_equal(self.solver._population_trial, expected_population_trial)

    def test_trims_when_outside_domain(self):
        self.solver._trim_population_trial_to_domain()
        np.testing.assert_array_less(self.solver._population_trial,
                                     5 * np.ones(self.solver._population_trial.shape) + 1e-14)
        np.testing.assert_array_less(-5 * np.ones(self.solver._population_trial.shape) - 1e-14,
                                     self.solver._population_trial)

    def test_trims_when_necessary(self):
        population_trial_org = self.solver._population_trial
        were_in_domain = np.logical_and(population_trial_org >= -5,
                                        population_trial_org <= 5).all(axis=1)
        self.solver._trim_population_trial_to_domain()

        np.testing.assert_array_equal(self.solver._population_trial[were_in_domain],
                                      population_trial_org[were_in_domain])

    def test_trims_linearly(self):
        original_delta_x = self.solver._population_trial - self.solver._population
        self.solver._trim_population_trial_to_domain()
        new_delta_x = self.solver._population_trial - self.solver._population

        scales = np.true_divide(new_delta_x, original_delta_x, where=original_delta_x != 0,
                                out=-np.ones((self.solver.population_size, self.solver.function.dimension)))

        out = [None] * self.solver.population_size
        for i, scale in enumerate(scales):
            ordinary_scale = scale[scale != -1]
            out[i] = np.max(np.abs(ordinary_scale - ordinary_scale[0]))
        np.testing.assert_allclose(out, np.zeros(self.solver.population_size), atol=1e-8)

    def test_trims_as_little_as_possible(self):
        were_not_in_domain = np.logical_or(self.solver._population_trial <= -5,
                                           self.solver._population_trial >= 5).any(axis=1)
        self.solver._trim_population_trial_to_domain()
        min_dist_from_edge_of_domain = np.minimum((5 - self.solver._population_trial[were_not_in_domain]).min(axis=1),
                                                  (self.solver._population_trial[were_not_in_domain] + 5).min(axis=1))
        np.testing.assert_allclose(min_dist_from_edge_of_domain, 0, atol=1e-14)


class TestAbstractDESetUpProcessOptimization(TestAbstractDESetUp):
    def test_crossover(self):
        population_trial_org = self.set_get_example_population_trial()
        self.solver._CR = np.ones(shape=self.solver.population_size)
        self.solver._crossover()
        np.testing.assert_array_equal(self.solver._population_trial,
                                      population_trial_org)  # whole of population_trial_org should go, cos all CR = 1

        population_trial_org = self.set_get_example_population_trial()
        self.solver._CR = np.zeros(shape=self.solver.population_size)
        self.solver._crossover()
        np.testing.assert_array_equal((self.solver._population_trial == population_trial_org).sum(axis=1),
                                      np.ones(self.solver.population_size))  # in every row, exactly one equal

    def test_evaluate(self):
        population_trial_org = self.set_get_example_population_trial()
        self.solver._evaluate()
        np.testing.assert_array_equal(self.solver._population_trial_f_value,
                                      self.restrigin5D(population_trial_org))

    def test_set_delta_f_and_get_improvement_bool(self):
        self.solver._population_f_value = np.ones(self.solver.population_size)
        self.solver._population_trial_f_value = np.linspace(0, 2, num=self.solver.population_size)
        have_improved = self.solver._set_delta_f_and_get_improvement_bool()

        desired_delta_f = np.where(self.solver._population_trial_f_value <= 1, 1.0 - self.solver._population_trial_f_value, 0)
        np.testing.assert_allclose(self.solver._delta_f, desired_delta_f)

        np.testing.assert_array_equal(have_improved, self.solver._population_trial_f_value <= 1)



if __name__ == '__main__':
    unittest.main()
