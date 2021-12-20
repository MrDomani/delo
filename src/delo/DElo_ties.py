import numpy as np
from .DElo import DElo
from .DistributionUtilities import edited_quantile


class DElo_ties(DElo):
    def __init__(self, population_size, p_best_rate=0.2, use_archive=True, archive_size=50,
                 portion_of_top_players=0.2, players_amount=100,
                 player_elo_rating_rate=0.4, task_elo_rating_rate=0.4,
                 history_for_ties_size=20, win_tie=0.5, tie_loss=0.0,
                 restart_eps_x=None, restart_eps_y=None,
                 variation_for_CR=0.1, scale_for_F=0.1,
                 logger=None, **logger_kwargs):
        """
        Modification of DElo algorithm with additional ties as a result of the game. Normal DElo is a special case of
        this version with win_tie=0 and win_loss=0.

        # TODO(history_for_ties=20, win_tie=0.02, tie_loss=0.1)
        """
        super().__init__(population_size=population_size,
                         p_best_rate=p_best_rate,
                         use_archive=use_archive,
                         archive_size=archive_size,
                         portion_of_top_players=portion_of_top_players,
                         players_amount=players_amount,
                         player_elo_rating_rate=player_elo_rating_rate,
                         task_elo_rating_rate=task_elo_rating_rate,
                         restart_eps_x=restart_eps_x,
                         restart_eps_y=restart_eps_y,
                         variation_for_CR=variation_for_CR,
                         scale_for_F=scale_for_F,
                         logger=logger,
                         **logger_kwargs)

        self.history_for_ties_size = history_for_ties_size
        self._history_for_ties_index = 0  # when this index exceed history_for_ties_size, it is set back to 0
        self.win_tie = win_tie
        self.tie_loss = tie_loss
        self._history_for_ties = None

        self.logger.DElo_ties_init(self._history_for_ties, self.win_tie, self.tie_loss)

    def optimize(self, described_function=None, max_f_evals=1000, print_every=None,
                 restarts_handled_externally=False, rng_seed=None):
        self._history_for_ties = np.zeros((self.history_for_ties_size, self.population_size))
        super().optimize(described_function=described_function, max_f_evals=max_f_evals, print_every=print_every,
                         restarts_handled_externally=restarts_handled_externally, rng_seed=rng_seed)

    def _selection(self):
        f_difference = self._population_f_value - self._population_trial_f_value  # we want this to be positive
        self._history_for_ties[self._history_for_ties_index] = f_difference
        self._history_for_ties_index += 1
        if self._history_for_ties_index == self.history_for_ties_size:
            self._history_for_ties_index = 0
        self.delta_f = f_difference * (f_difference > 0)

        indices_for_swap = (f_difference >= 0)  # True is a win for a player
        self.number_of_improvements += sum(indices_for_swap)

        self.logger.indices_for_swap(f_difference, self.delta_f, indices_for_swap)

        # ties
        positive_history_for_ties = self._history_for_ties[self._history_for_ties >= 0]  # one dimensional np.array
        win_tie_threshold = edited_quantile(positive_history_for_ties, self.win_tie)

        negative_history_for_ties = self._history_for_ties[self._history_for_ties < 0]  # one dimensional np.array
        tie_loss_threshold = edited_quantile(negative_history_for_ties, 1-self.tie_loss)  # self.tie_loss == 0.2 means 20% of losses will be ties

        ties = np.ones(len(self.delta_f), dtype="bool")
        ties[self.delta_f > win_tie_threshold] = False
        ties[self.delta_f < tie_loss_threshold] = False


        self._update_archive(indices_for_swap)
        self._population[indices_for_swap] = self._population_trial[indices_for_swap]
        self._population_f_value[indices_for_swap] = self._population_trial_f_value[indices_for_swap]

        self.logger.population(self._population, self._population_f_value)

        expected_results = self._calculate_expected_results()
        actual_results = indices_for_swap.astype(float)
        actual_results[ties] = 1/2
        self._update_elo_ratings(expected_results, actual_results)
        self._set_p_best()
        self._update_solution()





