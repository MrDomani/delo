import unittest
from delo import DElo, DescribedFunction
import numpy as np
from scipy.special import expit
from copy import copy

def restrigin5D(x):
    dim = 5
    if len(x.shape) == 1:
        raise IOError(f"x is 1D vector, but should be a 2D matrix: x = {x}")
    if x.shape[1] != dim:  # x is np.array with dim columns
        raise IOError(f"there should be {dim} columns of x, but: x = {x}")
    return 10 * dim + np.sum(np.power(x, 2) - 10 * np.cos(2 * np.pi * x), axis=1)

class TestInitialisation(unittest.TestCase):
    def setUp(self):
        self.delo = DElo(10)

    def test_init(self):
        self.assertIsInstance(self.delo, DElo)

    def test_init_players(self):
        self.assertEqual(self.delo.players.id.size, 100)
        self.assertEqual(self.delo.players.CR.size, 100)
        self.assertEqual(self.delo.players.F.size, 100)
        self.assertEqual(self.delo.players.rating.size, 100)
        desired = np.array([[0,1,2,3], [0,0,1,1], [0,1,0,1], [0,0,0,0]])
        self.delo._initialize_players(4)
        np.testing.assert_equal(self.delo.players.to_numpy(), desired)

class TestMutationMethods(unittest.TestCase):
    def setUp(self):
        described_function = DescribedFunction(restrigin5D,
                                               dimension=5,
                                               domain_lower_limit=-5,
                                               domain_upper_limit=5)

        self.delo = DElo(population_size=5, players_amount=25)
        # Initialize, but don't run loop
        self.delo.optimize(described_function, max_f_evals=-1)

    def test_draw_M_CR_and_M_F(self):
        self.delo.players.rating = np.zeros(self.delo.players.rating.shape)
        self.delo.portion_of_top_players = 0.2
        CRs, Fs = self.delo._draw_M_CR_and_M_F()
        centralized_CRs = CRs - CRs.mean()
        centralized_Fs = Fs - Fs.mean()
        self.assertTrue(np.any(centralized_CRs != np.zeros(centralized_CRs.shape)))
        self.assertTrue(np.any(centralized_Fs != np.zeros(centralized_Fs.shape)))

    def test_update_selected_players(self):
        self.delo._F = np.array([0.1, 0.3, 0.6, 0.9, 0.4])
        self.delo._CR = np.array([0.1, 0.3, 0.6, 0.9, 0.1])
        self.delo._update_selected_players()
        desired = np.array([0,6,12,24,2])
        self.assertEqual(self.delo._indices_of_selected_players.size, 5)
        np.testing.assert_equal(self.delo._indices_of_selected_players, desired)

    def test_generate_M_CR_and_M_F(self):
        self.delo.players.rating[self.delo.players.CR == 0] = 100
        self.delo.portion_of_top_players = 0.2
        CRs, Fs = self.delo._draw_M_CR_and_M_F()
        np.testing.assert_equal(CRs, np.zeros(5))

class TestSelectionMethods(unittest.TestCase):
    def setUp(self):
        described_function = DescribedFunction(restrigin5D,
                                               dimension=5,
                                               domain_lower_limit=-5,
                                               domain_upper_limit=5)

        self.delo = DElo(population_size=5, players_amount=25, player_elo_rating_rate=1, task_elo_rating_rate=1)
        # Initialize, but don't run loop
        self.delo.optimize(described_function, max_f_evals=-1, rng_seed=2021)
        self.delo._task_ratings = np.array([-3.0, -3.0, 0.0, 1.0, 2.0])
        self.delo._prepare_for_generation_processing()
        self.delo._mutate()
        self.delo._crossover()
        self.delo._evaluate()
        f_difference = self.delo._population_f_value - self.delo._population_trial_f_value
        self.delo._indices_for_swap = (f_difference >= 0)

    def test_calculate_expected_results(self):
        success_odds = self.delo._calculate_expected_results()
        np.testing.assert_equal(success_odds, expit(-self.delo._task_ratings))

    def test_update_ratings(self):
        old_task_ratings = copy(self.delo._task_ratings)
        expected_results = expit(-self.delo._task_ratings)  # players' rating is all 0
        actual_results = np.zeros(5)
        # W tym wypadku (ratingi równe 0) to nie ma znaczenia
        self.delo._indices_of_selected_players = np.array([0, 0, 0, 1, 2])
        self.delo._update_elo_ratings(expected_results, actual_results)
        np.testing.assert_allclose(self.delo._task_ratings - old_task_ratings, expected_results, atol=1e-14)
        desired_updates = np.zeros(25)
        desired_updates[0] = -expit(-old_task_ratings[0]) - expit(-old_task_ratings[1]) - expit(-old_task_ratings[2])
        desired_updates[1] = -expit(-old_task_ratings[3])
        desired_updates[2] = -expit(-old_task_ratings[4])
        np.testing.assert_allclose(desired_updates, self.delo.players.rating)
        #np.testing.assert_allclose(desired_updates, self.delo.players['rating'].to_numpy())

class TestOther(unittest.TestCase):
    def setUp(self):
        self.described_function = DescribedFunction(restrigin5D,
                                               dimension=5,
                                               domain_lower_limit=-5,
                                               domain_upper_limit=5)

    def test_rng_seed(self):
        delo_args = {'players_amount' : 25,
                     'population_size' : 20}
        max_f_evals = 4 * 5 * 10
        self.delo_1 = DElo(**delo_args)
        self.delo_1.optimize(self.described_function, max_f_evals=max_f_evals, rng_seed=2021)

        self.delo_2 = DElo(**delo_args)
        self.delo_2.optimize(self.described_function, max_f_evals=max_f_evals, rng_seed=2021)

        np.testing.assert_allclose(self.delo_1.get_solution()[0], self.delo_2.get_solution()[0], atol=1e-14)
        self.assertEqual(self.delo_1.get_solution()[1], self.delo_2.get_solution()[1])
        np.testing.assert_allclose(self.delo_1._population, self.delo_2._population, atol=1e-14)
        np.testing.assert_allclose(self.delo_1.players.to_numpy(), self.delo_2.players.to_numpy(), atol=1e-14)
        np.testing.assert_allclose(self.delo_1._task_ratings, self.delo_2._task_ratings)


if __name__ == '__main__':
    unittest.main()
