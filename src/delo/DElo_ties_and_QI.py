from .DElo import DElo
import numpy as np
from .DistributionUtilities import edited_quantile

class DElo_ties_and_QI(DElo):
    """
    Modification of DElo algorithm - Differential Evolution with Elo ranking system

    Optimization algorithm from Differential Evolution family. F and CR parameters are adjusted through optimizing
    using Elo ranking system. Updates in rating are based on relative change of function value and on loss-tie-win odds.
    Utilized mutation strategy: p-best. Succesful members from past will be stored in archive.
    Restart condition: minimum absolute dispersion across dimension.

    Notes
    -----
    A history of several (`history_for_ties_size`) previous function differences is recorded.
    Whenever a positive new function difference is less than `win_tie` percentile of previous function differences, it is a tie. Otherwise - a win.
    Whenever a negative new function difference is less than `tie_loss` percentile of history, it is a tie. Otherwise - a loss.

    Elo ratings, apart from predicting victory odds, are used to predict relative function difference.
    The prediction function is (rating_difference * `expectation_factor`).

    Examples
    --------
    >>> # Optimize quadratic function in 2D
    >>> import delo
    >>> import numpy as np
    >>>
    >>> def square(x):
    ...     return np.sum(x ** 2, axis=1)
    >>>
    >>> described_function = delo.DescribedFunction(square, dimension=2, domain_lower_limit=-10, domain_upper_limit=10)
    >>> algorithm = delo.DElo_ties_and_QI(100)
    >>>
    >>> solution, best_f_value = algorithm.optimize(described_function)
    >>> print(solution, best_f_value)
    0.0, 0.0

    >>> # If one have a function that takes a single argument and returns a single value, one have to wrap it like this:
    >>> import delo
    >>> import numpy as np
    >>>
    >>> def my_single_argument_function(x):
    ...     return np.sum(x ** 2)
    >>>
    >>> def my_multi_argument_wrapping(x):
    ...     return np.array([my_single_argument_function(xi) for xi in x])
    >>>
    >>> described_my_function = delo.DescribedFunction(my_multi_argument_wrapping,
    ...                                                dimension=5,
    ...                                                domain_lower_limit=-5,
    ...                                                domain_upper_limit=5)
    >>> algorithm = delo.DElo_ties_and_QI(100)
    >>>
    >>> solution, best_f_value = algorithm.optimize(described_my_function, max_f_evals=10000)
    >>> print(solution, best_f_value)
    [0.0 -0.0 -0.0  0.0  0.0], 1.1e-11
    """
    def __init__(self, population_size, p_best_rate=0.2, use_archive=True, archive_size=50,
                 portion_of_top_players=0.2, players_amount=100,
                 player_elo_rating_rate=0.4, player_elo_rating_rate_MOV=0.4, task_elo_rating_rate=0.4, task_elo_rating_rate_MOV=0.4, 
                 history_for_ties_size=20, win_tie=0.5, tie_loss=0.0,
                 expectation_factor=0.1, restart_eps_x=None, restart_eps_y=None,
                 variation_for_CR=0.1, scale_for_F=0.1,
                 logger=None, **logger_kwargs):
        """
        Initialise the algorithm, but not run in yet (see `optimize`).

        Parameters
        ----------
        population_size : positive int
        p_best_rate : float from (0,1]
            Fraction of members chosen in p_best mutation strategy.
        portion_of_top_players : float from (0, 1]
            Fraction of top players to use as starting values in mutation.
        players_amount : positive int
            How many players will be created. Should be a square of natural number.
        player_elo_rating_rate : non-negative float
            Constant used to calculate players' rating update based on predicted victory odds and observed effect.
        player_elo_rating_rate_MOV : non-negative float
            Constant used to calculate players' rating update based on expected and observed relative function difference.
        task_elo_rating_rate : non-negative float
            Constant used to calculate tasks' rating update based on predicted victory odds and observed effect.
        task_elo_rating_rate_MOV : non-negative float
            Constant used to calculate tasks' rating update based on expected and observed relative function difference.
        history_for_ties_size : positive int
            Function differences of previous `history_for_ties_size` will be recorded and used to determine win/tie/loss.
        win_tie : float from [0,1]
            Whenever a positive new function difference is less than `win_tie` percentile of previous function differences, it is a tie. Otherwise - a win.
        tie_loss : float from [0,1]
            Whenever a negative new function difference is less than `tie_loss` percentile of history, it is a tie. Otherwise - a loss.
        expectation_factor : non-negative float
            Used to calculate expected relative function difference as (rating_difference * `expectation_factor`).
        restart_eps_x : float, optional.
            Minimal acceptable absolute distance between members. If smaller, a restart occurs.
            If None, this condition will not be used.
        restart_eps_y : float, optional.
            Minimal acceptable absolute difference between function values. If smaller, a restart occurs.
            If None, this condition will not be used.
        """
        super().__init__(population_size=population_size, p_best_rate=p_best_rate, use_archive=use_archive,
                         archive_size=archive_size, portion_of_top_players=portion_of_top_players, players_amount=players_amount,
                         player_elo_rating_rate=player_elo_rating_rate, task_elo_rating_rate=task_elo_rating_rate,
                         restart_eps_x=restart_eps_x, restart_eps_y=restart_eps_y,
                         variation_for_CR=variation_for_CR, scale_for_F=scale_for_F,
                         logger=logger, **logger_kwargs)
        self.expectation_factor=expectation_factor
        self.task_elo_rating_rate_MOV = task_elo_rating_rate_MOV
        self.player_elo_rating_rate_MOV = player_elo_rating_rate_MOV
        
        self.history_for_ties_size = history_for_ties_size
        self.history_for_ties_index = 0  # when this index exceed history_for_ties_size, it is set back to 0
        self.win_tie = win_tie
        self.tie_loss = tie_loss
        self.history_for_ties = None

        self.logger.joint_init(self.history_for_ties_size, self.win_tie, self.tie_loss,
                               expectation_factor, player_elo_rating_rate_MOV, task_elo_rating_rate_MOV)
        
    def optimize(self, described_function=None, max_f_evals=1000, print_every=None,
                 restarts_handled_externally=False, rng_seed=None):
        self.history_for_ties = np.zeros((self.history_for_ties_size, self.population_size))
        return super().optimize(described_function=described_function, max_f_evals=max_f_evals, print_every=print_every,
                                restarts_handled_externally=restarts_handled_externally, rng_seed=rng_seed)
                         
    def _set_delta_f_and_get_improvement_bool(self):
        """

        Returns
        -------
        bool np vector.
            True = f value of trial member is better that original
        """
        f_difference = self._population_f_value - self._population_trial_f_value  # we want this to be positive
        self.history_for_ties[self.history_for_ties_index] = f_difference
        self.history_for_ties_index += 1
        if self.history_for_ties_index == self.history_for_ties_size:
            self.history_for_ties_index = 0
        self.delta_f = f_difference * (f_difference > 0)

        have_improved = (f_difference >= 0)  # True is a win for a player
        self.number_of_improvements += sum(have_improved)

        self.logger.indices_for_swap(f_difference, self.delta_f, have_improved)
        return have_improved
                         
    def _calculate_expected_results(self):
        rating_differences = self.players.rating[self._indices_of_selected_players] - self._task_ratings
        victory_odds = super()._calculate_expected_results()
        rating_differences[rating_differences < 0] = 0.0
        rating_differences[rating_differences > 1] = 1.0
        expected_relative_difference = rating_differences * self.expectation_factor
        return {'victory': victory_odds, 
                'relative_difference': expected_relative_difference}
        
    def _calculate_actual_results(self, have_improved):
        """

        Returns
        -------
        If delta_f == 0, then 0 is returned
        If delta_f != 0 and f_value == 0, then 1 is returned
        If delta_f/f_value > 1, then the result is truncated to 1
        """
        # ties
        positive_history_for_ties = self.history_for_ties[self.history_for_ties >= 0]  # one dimensional np.array
        win_tie_threshold = edited_quantile(positive_history_for_ties, self.win_tie)

        negative_history_for_ties = self.history_for_ties[self.history_for_ties < 0]  # one dimensional np.array
        tie_loss_threshold = edited_quantile(negative_history_for_ties, 1-self.tie_loss)  # self.tie_loss == 0.2 means 20% of losses will be ties

        ties = np.ones(len(self.delta_f), dtype="bool")
        ties[self.delta_f > win_tie_threshold] = False
        ties[self.delta_f < tie_loss_threshold] = False
        
        loss_tie_win = have_improved.astype(float)
        loss_tie_win[ties] = 1/2
        
        # margin of victory
        out = (self.delta_f != 0).astype(np.float64)
        actual_relative_difference = np.true_divide(self.delta_f, np.abs(self._population_f_value), out=out, where=self._population_f_value != 0)
        actual_relative_difference[actual_relative_difference > 1] = 1
        return {'victory':loss_tie_win, 
                'relative_difference': actual_relative_difference}
        
    def _update_elo_ratings(self, expected_results, actual_results):
        update = self.player_elo_rating_rate * (actual_results['victory'] - expected_results['victory']) + \
                 self.player_elo_rating_rate_MOV * (actual_results['relative_difference'] - expected_results['relative_difference'])

        # note, that if one player plays more than one in the generation,
        # its rating from the beginning of the generation is taken into calculations
        player_update = np.zeros(self.players.number_of_players)
        for i in range(self.population_size):  # len(self.indexes_of_selected_players) == self.population_size
            player_update[self._indices_of_selected_players[i]] += update[i]
        self.players.rating += player_update  # it is 0 for unselected players

        task_updates = self.task_elo_rating_rate * (expected_results['victory'] - actual_results['victory']) + \
                       self.task_elo_rating_rate_MOV * (expected_results['relative_difference'] - actual_results['relative_difference'])
        self._task_ratings += task_updates

        self.logger.joint_elo_ratings(expected_results['victory'], actual_results['victory'],
                                      expected_results['relative_difference'], actual_results['relative_difference'],
                                      player_update, self.players.rating, task_updates, self._task_ratings)