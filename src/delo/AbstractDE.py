from abc import ABC, abstractmethod
from .DescribedFunction import DescribedFunction
from .Logger import *
from .DistributionUtilities import *
from .CustomExceptions import *


class AbstractDE(ABC):
    """Abstact class for SHADE and DElo

    Optimization algorithm from Differential Evolution family. F and CR parameters are adjusted through optimizing.
    Utilized mutation strategy: p-best. Succesful members from past will be stored in archive.
    """
    def __init__(self, population_size, p_best_rate=0.2, use_archive=True, archive_size=50,
                 restart_eps_x=None, restart_eps_y=None, variation_for_CR=0.1, scale_for_F=0.1, logger=None, **logger_kwargs):
        """Initialise the algorithm, but not run in yet (see `optimize`).

        Parameters
        ----------
        population_size : positive int
        p_best_rate : float from (0,1]
            Fraction of members chosen in p_best mutation strategy.
        restart_eps_x : float, optional.
            Minimal acceptable absolute distance between members. If smaller, a restart occurs.
            If None, this condition will not be used.
        restart_eps_y : float, optional.
            Minimal acceptable absolute difference between function values. If smaller, a restart occurs.
            If None, this condition will not be used.           
        """
        self.function = None

        self._check_init_correctness(restart_eps_x, restart_eps_y, use_archive, archive_size,
                                     population_size, p_best_rate)

        self.restart_eps_x = restart_eps_x  # could be None
        self.restart_eps_y = self.restart_eps_x if (restart_eps_y is None) else restart_eps_y

        self.use_archive = use_archive
        self.max_archive_size = archive_size

        self.population_size = population_size
        self.p_best_rate = p_best_rate
        self.number_of_best = int(max(np.floor(self.p_best_rate * self.population_size),
                                      3))  # at least 3, so that range [2, number_of_best) will have at least one integer
        self._remaining_evals = 0
        self._generations_processed = 0

        self._best_member_ever = None
        self._best_f_value_ever = None

        self.variation_for_CR = variation_for_CR
        self.scale_for_F = scale_for_F

        self._number_of_improvements = 0

        self._process_logger_init_args(logger_kwargs, logger)
        self.logger.AbstractDE_init(self.restart_eps_x, self.restart_eps_y, self.use_archive,
                                    self.max_archive_size, self.population_size, self.p_best_rate,
                                    self.variation_for_CR, self.scale_for_F)

    def _check_init_correctness(self, restart_eps_x, restart_eps_y, use_archive, max_archive_size,
                                population_size, p_best_rate):
        if restart_eps_x is not None and restart_eps_x <= 0:
            raise ImproperRestartEpsilonException(
                "Provided restart epsilon for x is improper. It should be None or positive float")
        if restart_eps_y is not None and restart_eps_y <= 0:
            raise ImproperRestartEpsilonException(
                "Provided restart epsilon for y is improper. It should be None or positive float")

        if use_archive and max_archive_size.__class__.__name__ != "int":
            raise NonIntArchiveException()
        if use_archive and max_archive_size < 0:
            raise NegativeArchiveException()

        if population_size.__class__.__name__ != "int":
            raise PopulationSizeNotIntException(
                f"Provided size for population is not integer, but {population_size.__class__.__name__}, namely, {population_size}")
        if population_size <= 0:
            raise NonPositivePopulationSizeException()
        if population_size == 1:
            raise PopulationSize1Exception()

        if p_best_rate < 0 or p_best_rate > 1:
            raise p_best_rateOutOf01Exception(
                f"Provided p_best_rate is {p_best_rate}, which is out of [0,1] range")

    def _process_logger_init_args(self, logger_kwargs, logger):
        if 'logger_str' in logger_kwargs:
            logger_kwargs.pop('logger_str')
        if logger_kwargs:
            print(logger_kwargs)
            warnings.warn('Providing arguments of logger via init is deprecated. Use `logger` argument instead')
            if logger is not None:
                raise Exception('Provided both `logger` arguments and logger_kwargs (deprecated). Provide only one')
            if not logger_kwargs['use_logger']:
                self.logger = FakeLogger()
            else:
                self.logger = Logger(logger_kwargs['logger_file'], what_to_log=logger_kwargs['what_to_log'])
            return
        if logger is None:
            self.logger=FakeLogger()
        else:
            self.logger = logger

    def get_solution(self):
        """Get solution found during optimization process

        Returns
        -------
        Tuple
            `solution` (member with lowest f-value), `best_f_value`.
        """
        return self._best_member_ever, self._best_f_value_ever

    def optimize(self, described_function, max_f_evals=1000, print_every=None, restarts_handled_externally=False,
                 rng_seed=None):
        """Optimize the described function.

        Pass the target function and start the optimization process.

        Parameters
        ----------
        described_function : DescribedFunction
            function to be optimized with attributes, created with DescribedFunction.
        print_every : int, optional
            Info about verbosity. Every `print_every` generation information about state of optimization
            will be printed on console.
        restarts_handled_externally: bool.
            If True and restarting conditions are met, the algorithm ends. If False
            and restarting conditions are met, the algorithm restarts.
        rng_seed : int, optional
            seed to be used in pseudorandom number generation. Same seed leads to same outcomes.

        Returns
        -------
        Tuple
            `solution` (member with lowest f-value), `best_f_value`.
        """
        self._prepare_optimization(described_function, max_f_evals, print_every, rng_seed)

        generations_done = 0
        generations_after_last_restart = 0
        while self._remaining_evals >= self.population_size:
            self.logger.start_generation(generations_done, generations_after_last_restart)
            if self.printer.generation(generations_done, generations_after_last_restart,
                                       self.current_best_f, self._best_f_value_ever,
                                       self._number_of_improvements):
                self._number_of_improvements = 0  # this will reset if information was printed
            self._process_generation()

            # restart condition
            if self._check_restart_condition():
                self.printer.restarting(generations_after_last_restart, self.current_best_f)
                self.logger.restarting(generations_after_last_restart, self.current_best_f)

                if restarts_handled_externally:
                    return  # external function can restart the algorithm
                self._restart_search()
                generations_after_last_restart = -1

            generations_done += 1
            generations_after_last_restart += 1

        self.logger.end_optimization(self._generations_processed, self._best_member_ever, self._best_f_value_ever,
                                     self.restarts)
        self.printer.optimizing_complete(self.restarts, generations_done, self._remaining_evals,
                                         self._generations_processed, self._best_f_value_ever)

        return self.get_solution()

    def _prepare_optimization(self, described_function, max_f_evals, print_every, rng_seed):
        self.logger.start_optimization(rng_seed)

        self.rng = np.random.default_rng(seed=rng_seed)

        # check correctness of input
        if not isinstance(described_function, DescribedFunction):
            self.logger.function_not_Describedfunction()
            raise Exception('`described_function` must be of Describedfunction class.')
        self.function = described_function

        self._remaining_evals = max_f_evals
        self._restart_search(initial=True)

        self.logger.optimization_preparation(max_f_evals, self.function.dimension, self.number_of_best)
        if print_every is None or int(print_every) < 1:
            self.printer = FakePrinter(print_every)
        else:
            self.printer = Printer(print_every)

        self.printer.start_optimization(self._remaining_evals + self.population_size,  # restart_search() already used some of remaining_evals
                                        rng_seed)

    def _restart_search(self, initial=False):
        if initial:
            self.restarts = 0
        else:
            self.logger.restart()
            self.restarts += 1
        self._clear_archive()
        self._reset_CR_and_F()
        self._init_population_and_reset_p_best()
        self._delta_f = np.zeros(self.population_size)
        self._number_of_improvements = 0

    def _clear_archive(self):
        if self.use_archive:
            self.archive = np.empty((0, self.function.dimension))  # just like self.population, but empty
        else:
            self.archive = None

        self.logger.archive(self.archive)

    @abstractmethod
    def _reset_CR_and_F(self):
        pass

    def _init_population_and_reset_p_best(self):
        self._init_population()
        self._set_p_best()
        self._update_solution()

    def _init_population(self):
        """
        Random initialization from uniform distribution with limits self.function.domain_lower_limit, self.function.domain_upper_limit
        """
        U = self.rng.random((self.population_size, self.function.dimension))
        self._population = self.function.domain_lower_limit + U * (
                self.function.domain_upper_limit - self.function.domain_lower_limit)
        self._population_trial = self._population.copy()

        self._remaining_evals -= self.population_size
        self._population_f_value = self.function.call(self._population)
        self._population_trial_f_value = self._population_f_value.copy()

        self.logger.population(self._population, self._population_f_value)

    def _set_p_best(self):
        scores_index_sorted = self._population_f_value.argsort()

        self.current_worst_i = scores_index_sorted[self.population_size - 1]
        self.current_worst_f = self._population_f_value[self.current_worst_i]  # it is used in check_restart_condition()

        self.current_p_best_i = scores_index_sorted[0:self.number_of_best]  # index of i-th best specimen
        self.current_p_best_f = self._population_f_value[self.current_p_best_i]  # score of i-th best specimen

        self.current_best_i = self.current_p_best_i[0]
        self.current_best_f = self.current_p_best_f[0]

        self.logger.p_best(scores_index_sorted, self.current_worst_i, self.current_worst_f,
                           self.current_p_best_i, self.current_p_best_f,
                           self.current_best_i, self.current_best_f)

    def _update_solution(self):
        """
        Compare best solution found since last restart to best known ever and update if better.
        If no solution is available, then it is set.
        Assuming, that self.current_best_* are up to date.
        """
        if self._best_f_value_ever is None or self.current_best_f < self._best_f_value_ever:
            self._best_member_ever = self._population[self.current_best_i]
            self._best_f_value_ever = self.current_best_f

            self.logger.update_solution(self._best_member_ever, self._best_f_value_ever)

    def _process_generation(self):
        """Process one generation (iteration) of optimizing process. """
        self._prepare_for_generation_processing()
        self._mutate()
        self._crossover()
        self._evaluate()
        self._selection()
        self._generations_processed += 1

    def _prepare_for_generation_processing(self):
        """Generate and set all parameters (Fs and CRs included) necessary for mutation"""
        self._delta_f = np.zeros(self.population_size)
        self._set_CR_and_F()

    def _set_CR_and_F(self):
        drawn_M_CR, drawn_M_F = self._draw_M_CR_and_M_F()

        # CR is chopped Normal:
        self._CR = chopped_normal(rng=self.rng, size=self.population_size, location=drawn_M_CR,
                                  variation=self.variation_for_CR)  # from DistributionUtilities.py

        # F is chopped Cauchy:
        self._F = chopped_cauchy(rng=self.rng, size=self.population_size, location=drawn_M_F,
                                 scale=self.scale_for_F)  # from DistributionUtilities.py

        self.logger.drawn_CR_F(drawn_M_CR, drawn_M_F, self._CR, self._F)

    @abstractmethod
    def _draw_M_CR_and_M_F(self):
        pass

    def _mutate(self):
        r1, r2 = self._choose_indices_for_mutation()
        x_r1, x_r2 = self._get_members_for_mutation(r1, r2)

        self.logger.members_for_mutation(r1, r2, x_r1, x_r2)

        x_p_best = self._choose_p_best_members_for_mutation()
        self._population_trial = self._population + ((x_p_best - self._population + x_r1[:, :] - x_r2[:, :]).T * self._F).T
        self._trim_population_trial_to_domain()

        self.logger.population_trial(x_p_best, self._population_trial)

    def _choose_indices_for_mutation(self):
        if self.use_archive and self.archive.shape[0] != 0:  # if it is 0, archive is empty
            archive_size = self.archive.shape[0]
        else:
            archive_size = 0

        return choose_2_columns_of_integers(rng=self.rng, nrow=self.population_size, matrix_of_restrictions=np.array(
            [[0, 0], [self.population_size, self.population_size+archive_size]]))

    def _get_members_for_mutation(self, indices1, indices2):
        """

        Parameters
        ----------
        indices1 : 1-D np.ndarray
            Indices from population.
        indices2 : 1-D np.ndarray
            Indices from population and archive. If indices2 exceed population_size, it will be drawn from archive.
        """
        x_r1 = self._population[indices1]
        x_r2 = np.empty((self.population_size, self.function.dimension))

        use_archive = indices2 >= self.population_size
        use_population = ~(use_archive)
        x_r2[use_population] = self._population[indices2[use_population]]  # population
        x_r2[use_archive] = self.archive[indices2[use_archive] - self.population_size]  # archive

        return x_r1, x_r2

    def _choose_p_best_members_for_mutation(self):
        numbers_of_specimens_to_choose_from = self.rng.integers(2, self.number_of_best,
                                                                self.population_size)  # note that numbers_of_specimens_to_choose_from is of length population_size and: 2 <= numbers_of_specimens_to_choose_from < self.number_of_best

        p_best_members_indices = self.current_p_best_i[self.rng.integers(numbers_of_specimens_to_choose_from)]

        self.logger.p_best_draw(numbers_of_specimens_to_choose_from, p_best_members_indices)

        p_best_members = self._population[p_best_members_indices]
        return p_best_members

    def _trim_population_trial_to_domain(self):
        """
        For every member with trial vector outside of the domain
        find a new trial vector located on straight line between trial vector and original vector
        that is on the edge of the domain
        And set it as new trial vector
        """
        are_not_in_domain=np.logical_or(self._population_trial <= self.function.domain_lower_limit,
                                        self._population_trial >= self.function.domain_upper_limit).any(axis=1)
        if not np.any(are_not_in_domain):
            return
        are_not_in_domain=np.nonzero(are_not_in_domain)
        delta_xs = self._population_trial[are_not_in_domain] - self._population[are_not_in_domain]

        # We have to satisfy the following inequality:
        #     domain_lower_limit <= population[i,j] + scale_candidates[i,j]*delta_xs[i, j] <= domain_upper_limit
        # population[i,j] is between domain limits, scale_candidates[i,j] is between 0 and 1
        # so the only one side of inequality is relevant. Which? - that depends on sign of delta_x
        relevant_domain_limits=np.where(delta_xs>0, self.function.domain_upper_limit, self.function.domain_lower_limit)

        # now, after transformations, the inequality reduces to
        #     scale_candidates[i,j] <= (relevant_domain_limit - population[i,j]) / delta_xs[i,j]
        # we want scale_candidates[i,j] to be as large as possible, so we set it as equal to the right side
        scale_candidates=np.true_divide(relevant_domain_limits - self._population[are_not_in_domain], delta_xs,
                                        where=delta_xs!=0, out=np.ones(delta_xs.shape))
        scale = scale_candidates.min(axis=1, keepdims=True)
        # correction in case of floating point weird behaviour
        scale=np.minimum(scale, 1)
        self._population_trial[are_not_in_domain]= self._population[are_not_in_domain] + scale * delta_xs
        return

    def _crossover(self):
        replace_with_trial_coord = self._get_replacement_decisions_for_crossover()
        self._population_trial = np.where(replace_with_trial_coord, self._population_trial, self._population)

        self.logger.swap_population_trial(replace_with_trial_coord, self._population_trial)

    def _get_replacement_decisions_for_crossover(self):
        """

        Returns
        -------
        a bool ndarray of size self.function.dimension x self.population_size.
            True = choose coordinate from trial vector, False = keep coordinate from old vector
        """
        sure_swap_indices = self.rng.choice(self.function.dimension, size=self.population_size)
        draws = self.rng.random(size=(self.population_size, self.function.dimension))
        swap = draws < np.tile(self._CR.reshape((-1, 1)), self.function.dimension)
        swap[np.arange(self.population_size), sure_swap_indices] = True
        return swap

    def _evaluate(self):
        self._remaining_evals -= self.population_size
        self._population_trial_f_value = self.function.call(self._population_trial)

    @abstractmethod
    def _selection(self):
        pass

    def _set_delta_f_and_get_improvement_bool(self):
        """

        Returns
        -------
        bool np vector.
            True = f value of trial member is better that original
        """
        f_difference = self._population_f_value - self._population_trial_f_value  # we want this to be positive
        self._delta_f = f_difference * (f_difference > 0)
        have_improved = (f_difference >= 0)
        self.logger.indices_for_swap(f_difference, self._delta_f, have_improved)
        return have_improved

    def _process_evaluation_results(self, have_improved):  # it is small function in DE, but it is overwritten in DElo
        self._number_of_improvements += sum(have_improved)
        self._update_archive(have_improved)

    def _update_archive(self, indices_to_archive):
        """
        Assuming, that members were evaluated, but not swapped

        Parameters
        ----------
        indices_to_archive: bool np.array of length self.population_size
        """
        if not self.use_archive: return
        self.archive = np.concatenate((self.archive, self._population[indices_to_archive]))

        if self.archive.shape[0] > self.max_archive_size:
            number_to_delete = int(self.archive.shape[0] - self.max_archive_size)
            r = self.rng.choice(self.archive.shape[0], size=number_to_delete, replace=False)

            self.logger.remove_from_archive(r)

            self.archive = np.delete(self.archive, r, 0)  # 0 means delete rows, not columns

        self.logger.archive(self.archive)

    def _replace_with_improved_members(self, have_improved):
        self._population[have_improved] = self._population_trial[have_improved]
        self._population_f_value[have_improved] = self._population_trial_f_value[have_improved]
        self.logger.population(self._population, self._population_f_value)
        self._set_p_best()
        self._update_solution()

    @abstractmethod
    def _check_restart_condition(self):
        pass
