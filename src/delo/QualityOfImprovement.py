from .DElo import DElo
import numpy as np


class DElo_QualityOfImprovement(DElo):

    def __init__(self, population_size, p_best_rate=0.2, use_archive=True, archive_size=50,
                 portion_of_top_players=0.2, players_amount=100,
                 player_elo_rating_rate=0.4, task_elo_rating_rate=0.4, expectation_factor=0.1,
                 restart_eps_x=None, restart_eps_y=None,
                 variation_for_CR=0.1, scale_for_F=0.1,
                 logger=None, **logger_kwargs):
        """
        Differential Evolution with Elo ranking system
        
        Updates in rating are based on relative change of function value.
        
        :arg expectation_factor: The expected difference in Elo is returned as (Rating_diff) * `expectation_factor`.
        """
        super().__init__(population_size=population_size, p_best_rate=p_best_rate, use_archive=use_archive,
                         archive_size=archive_size, portion_of_top_players=portion_of_top_players, players_amount=players_amount,
                         player_elo_rating_rate=player_elo_rating_rate, task_elo_rating_rate=task_elo_rating_rate,
                         restart_eps_x=restart_eps_x, restart_eps_y=restart_eps_y,
                         variation_for_CR=variation_for_CR, scale_for_F=scale_for_F,
                         logger=logger, **logger_kwargs)
        self.expectation_factor=expectation_factor
        self.logger.log('expectation_factor', expectation_factor, False)
                         
    def _calculate_expected_results(self):
        rating_differences = self.players.rating[self._indices_of_selected_players] - self._task_ratings
        rating_differences[rating_differences < 0] = 0.0
        rating_differences[rating_differences > 1] = 1.0
        expected_relative_difference = rating_differences * self.expectation_factor
        return expected_relative_difference
        
    def _calculate_actual_results(self, have_improved):
        """
        Jeśli delta_f = 0, to zwracamy zawsze 0
        Jeśli delta_f != 0 i f_value = 0. to zwracamy 1
        Jeśli delta_f/f_value > 1, to ścinamy do 1
        """
        out = (self._delta_f != 0).astype(np.float64)
        actual_results = np.true_divide(self._delta_f, np.abs(self._population_f_value), out=out, where=self._population_f_value != 0)
        actual_results[actual_results > 1] = 1
        return actual_results    