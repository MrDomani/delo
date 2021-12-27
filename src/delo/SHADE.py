import numpy as np
from .AbstractDE import AbstractDE
from .Logger import FakeLogger


class SHADE(AbstractDE):
    """

    Optimization algorithm from Differential Evolution family. F and CR parameters are adjusted through optimizing
    using sucess history. Utilized mutation strategy: p-best. Succesful members from past will be stored in archive.
    Restart condition: minimum relative dispersion across dimension.
    """
    def __init__(self, population_size, p_best_rate=0.2, use_archive=True, archive_size=50,
                 history_size=100, restart_eps_x=None, restart_eps_y=None,
                 initial_M_CR=0.5, initial_M_F=0.5,
                 logger=None, **logger_kwargs):
        """
        Succcess-History based parameter Adaptation for Differential Evolution

        Initialise the algorithm, but not run in yet (see `optimize`).

        Parameters
        ----------
        population_size : positive int
        p_best_rate : float from (0,1]
            fraction of members chosen in p_best mutation strategy
        restart_eps_x : positive float, optional
            minimal acceptable relative distance between members. If smaller, a restart occurs
        restart_eps_y : positive float, optional
            minimal acceptable relative difference between function values. If smaller, a restart occurs
        initial_M_F : float from [0, 1]
        initial_M_CR : float from [0, 1]
        """
        super().__init__(population_size=population_size,
                         p_best_rate=p_best_rate,
                         use_archive=use_archive,
                         archive_size=archive_size,
                         restart_eps_x=restart_eps_x,
                         restart_eps_y=restart_eps_y,
                         logger=logger,
                         **logger_kwargs)

        self.history_size = history_size
        self.initial_M_CR = initial_M_CR
        self.initial_M_F = initial_M_F

        self.logger.SHADE_init(self.history_size, self.initial_M_CR, self.initial_M_F)

    def _reset_CR_and_F(self):
        self.k = -1 # index for memory of CR and F; It will start at 0, because it is increased in the beginning of generation() function
        self.M_CR = np.ones(self.history_size) * self.initial_M_CR
        self.M_F = np.ones(self.history_size) * self.initial_M_F
        self._CR = np.zeros(self.population_size)
        self._F = np.zeros(self.population_size)

    def _prepare_for_generation_processing(self):
        super()._prepare_for_generation_processing()
        self.k += 1  # index for memory of CR and F
        if self.k == self.history_size: self.k = 0

    def _draw_M_CR_and_M_F(self):
        r = self.rng.choice(self.history_size, size=self.population_size, replace=True)
        return self.M_CR[r], self.M_F[r]

    def _selection(self):
        have_improved = self._set_delta_f_and_get_improvement_bool()
        if sum(have_improved) == 0:
            self.logger.unsuccessful_generation()
            if isinstance(self.logger, FakeLogger):  # if we do log, we want to have logs of the proper size
                return  # this generation was unsuccessful

        self._process_evaluation_results(have_improved)
        self._replace_with_improved_members(have_improved)

    def _process_evaluation_results(self, have_improved):
        super()._process_evaluation_results(have_improved)
        self._update_M_CR_and_M_F()

    def _update_M_CR_and_M_F(self):
        S = sum(self._delta_f)

        if S > 0:  # S > 0 iff the set of S_{CF} is non-empty
            w = self._delta_f / S  # S is always non-negative, so here S is positive
            self.M_CR[self.k] = sum(w * self._CR)
            self.M_F[self.k] = sum((w * (self._F ** 2))) / sum((w * self._F))

            self.logger.new_CR_F(w, self._CR[w > 0], self.M_CR[self.k], self._F[w > 0], self.M_F[self.k])

        self.logger.updated_CR_F(self.M_CR, self.M_F)

    def _check_restart_condition(self):
        """
        Restart will occur in one of two cases:
        1) If relative distance between members is too small
        2) If relative difference between function values is too small

        Returns
        -------
        bool
            are restart conditions satisfied?
        """
        if self.restart_eps_x is None:
            return False

        cond_restart = False

        # RelMax
        if np.min(self._population.max(axis=0) - self._population.min(axis=0)) <= \
                self.restart_eps_x * np.abs(self._population).max():
            self.logger.restarting_cond_x(np.min(self._population.max(axis=0) - self._population.min(axis=0)),
                                          np.abs(self._population).max(),
                                          self.restart_eps_x, abs=True)
            cond_restart = True

        if (self.current_worst_f - self.current_best_f) <= \
                self.restart_eps_y * abs(self.current_worst_f):
            self.logger.restarting_cond_y((self.current_worst_f - self.current_best_f),
                                          abs(self.current_worst_f),
                                          self.restart_eps_y,
                                          abs=False)  # abs is False, cos it is relative, cos eps_y is multiplied by worst_f
            cond_restart = True

        return cond_restart

    


