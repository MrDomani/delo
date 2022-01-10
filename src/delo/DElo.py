import numpy as np
from .AbstractDE import AbstractDE
from scipy.spatial.distance import cdist
from scipy.special import expit
from warnings import warn
from .CustomExceptions import *


class DElo(AbstractDE):
    """Differential Evolution with Elo ranking system

    Optimization algorithm from Differential Evolution family. F and CR parameters are adjusted through optimizing
    using Elo ranking system. Utilized mutation strategy: ``p-best``. Succesful members from past will be stored in archive.
    Restart condition: minimum absolute dispersion across dimension.

    Examples
    --------
    >>> # Optimize quadratic function in 2D::
    >>> import delo
    >>> import numpy as np
    >>>
    >>> def square(x):
    ...     return np.sum(x ** 2, axis=1)
    >>>
    >>> described_function = delo.DescribedFunction(square, dimension=2, domain_lower_limit=-10, domain_upper_limit=10)
    >>> algorithm = delo.DElo(100)
    >>>
    >>> solution, best_f_value = algorithm.optimize(described_function)
    >>> print(solution, best_f_value)
    0.0, 0.0

    >>> # If one have a function that takes a single argument and returns a single value, one have to wrap it like this::
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
    >>> algorithm = delo.DElo(100)
    >>>
    >>> solution, best_f_value = algorithm.optimize(described_my_function, max_f_evals=10000)
    >>> print(solution, best_f_value)
    [0.0  0.0 -0.0  0.0  0.0], 8.5e-09
    """

    class Players:
        def __init__(self, number_of_players, id, CR, F, rating):
            self.number_of_players = number_of_players
            self._id = id
            self._CR = CR
            self._F = F
            self.rating = rating

        @property
        def id(self):
            return self._id

        @property
        def CR(self):
            return self._CR

        @property
        def F(self):
            return self._F

        def to_numpy(self):
            return np.array([self.id,
                             self.CR,
                             self.F,
                             self.rating])

    def __init__(self, population_size, p_best_rate=0.1, use_archive=True, archive_size=50,
                 portion_of_top_players=0.1, players_amount=100,
                 player_elo_rating_rate=0.4, task_elo_rating_rate=0.4,
                 restart_eps_x=None, restart_eps_y=None,
                 variation_for_CR=0.1, scale_for_F=0.1,
                 logger=None, **logger_kwargs):
        """Initialise the algorithm, but not run in yet (see `optimize`).

        Parameters
        ----------
        population_size : positive int
        p_best_rate : float from (0,1]
            Fraction of members chosen in ``p_best`` mutation strategy.
        portion_of_top_players : float from [1/population_size, 1]
            Fraction of top players to use as starting values in mutation.
        players_amount : positive int
            How many players will be created. Has to be a square of natural number.
        restart_eps_x : float, optional
            Minimal acceptable absolute distance between members. If smaller, a restart occurs. If ``None``, restarting will never occur.
        restart_eps_y : float, optional
            Minimal acceptable absolute difference between function values. If smaller, a restart occurs. If None, this will be same as ``restart_eps_x``.
        logger : Logger, PickleLogger, optional
            If provided, logs to the file information on the process of optimizing
        """
        super().__init__(population_size,
                         p_best_rate=p_best_rate,
                         use_archive=use_archive,
                         archive_size=archive_size,
                         restart_eps_x=restart_eps_x,
                         restart_eps_y=restart_eps_y,
                         variation_for_CR=variation_for_CR,
                         scale_for_F=scale_for_F,
                         logger=logger,
                         **logger_kwargs)

        self.number_of_improvements = 0

        if portion_of_top_players > 1 or portion_of_top_players < 1/players_amount:
            raise portion_of_top_playersImproperException()
        self.portion_of_top_players = portion_of_top_players

        self._task_ratings = np.zeros(population_size)

        self.player_elo_rating_rate = player_elo_rating_rate
        self.task_elo_rating_rate = task_elo_rating_rate

        self._initialize_players(players_amount)

        self.logger._DElo_init(portion_of_top_players, self.player_elo_rating_rate,
                               self.task_elo_rating_rate, self.players.number_of_players)

    def _initialize_players(self, players_amount):
        side_grid_length = int(np.sqrt(players_amount).round())
        actual_player_amount = side_grid_length ** 2
        if players_amount != actual_player_amount:
            self.logger._improper_player_amount(players_amount)
            warn('`players_amount` is not a square of natural number.')
        self.players = self.Players(number_of_players=actual_player_amount,
                                    id=np.arange(actual_player_amount),
                                    CR=np.repeat(np.linspace(0, 1, num=side_grid_length), side_grid_length),
                                    F=np.tile(np.linspace(0, 1, num=side_grid_length), side_grid_length),
                                    rating=np.zeros(actual_player_amount))

    def _reset_CR_and_F(self):
        """W SHADE - restart historii. W DElo - restart ratingów"""
        self.players.rating = np.zeros(self.players.number_of_players)
        self._task_ratings = np.zeros(self.population_size)
        self._CR = np.empty(self.population_size)
        self._F = np.empty(self.population_size)

    def _prepare_for_generation_processing(self):
        super()._prepare_for_generation_processing()
        self._update_selected_players()

    def _update_selected_players(self):
        indices = cdist(np.column_stack((self.players.CR, self.players.F)),
                        np.column_stack((self._CR, self._F))).argmin(axis=0)
        self._indices_of_selected_players = indices

        self.logger._indices_of_selected_players(self._indices_of_selected_players)

    def _draw_M_CR_and_M_F(self):
        quantile = 1-self.portion_of_top_players
        top_players_bool = self.players.rating >= np.quantile(self.players.rating, quantile)
        top_players_indexes = np.arange(self.players.number_of_players)[top_players_bool]
        r = self.rng.choice(top_players_indexes.shape[0], size=self.population_size, replace=True)

        self.logger._top_players(top_players_indexes, top_players_indexes[r])

        return self.players.CR[top_players_indexes[r]], self.players.F[top_players_indexes[r]]

    def _selection(self):
        have_improved = self._set_delta_f_and_get_improvement_bool()
        self._process_evaluation_results(have_improved)
        self._replace_with_improved_members(have_improved)

    def _process_evaluation_results(self, have_improved):
        super()._process_evaluation_results(have_improved)
        expected_results = self._calculate_expected_results()
        actual_results = self._calculate_actual_results(have_improved)
        self._update_elo_ratings(expected_results, actual_results)

    def _calculate_expected_results(self):
        rating_differences = self.players.rating[self._indices_of_selected_players] - self._task_ratings
        return expit(rating_differences)

    def _calculate_actual_results(self, have_improved):
        return have_improved.astype(int)

    def _update_elo_ratings(self, expected_results, actual_results):
        update = self.player_elo_rating_rate * (actual_results - expected_results)

        # note, that if one player plays more than one in the generation,
        # its rating from the beginning of the generation is taken into calculations
        player_update = np.zeros(self.players.number_of_players)
        for i in range(self.population_size):  # len(self.indexes_of_selected_players) == self.population_size
            player_update[self._indices_of_selected_players[i]] += update[i]
        self.players.rating += player_update  # it is 0 for unselected players

        task_updates = self.task_elo_rating_rate * (expected_results - actual_results)
        self._task_ratings += task_updates

        self.logger._elo_ratings(expected_results, actual_results, player_update,
                                 self.players.rating, task_updates, self._task_ratings)
    
    def _check_restart_condition(self):
        """
        Restart will occur in one of two cases:
        1) If absolute distance between members is too small
        2) If absolute difference between function values is too small

        Returns
        -------
        bool
            are restart conditions satisfied?
        """
        if self.restart_eps_x is None:
            return False
        cond_restart = False

        if np.min(self._population.max(axis=0) - self._population.min(axis=0)) <= self.restart_eps_x:
            self.logger._restarting_cond_x(np.min(self._population.max(axis=0) - self._population.min(axis=0)),
                                           np.abs(self._population).max(),
                                           self.restart_eps_x, abs=True)
            cond_restart = True
        if (self.current_worst_f - self.current_best_f) <= self.restart_eps_y: 
            self.logger._restarting_cond_y((self.current_worst_f - self.current_best_f),
                                           abs(self.current_best_f),
                                           self.restart_eps_y, abs=True)
            cond_restart = True

        return cond_restart






