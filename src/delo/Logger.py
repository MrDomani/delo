import logging
import time
import numpy as np
import re
import ast
from sys import maxsize
import warnings

np.set_printoptions(suppress=True,  # no scientific notation
                    threshold=maxsize,  # logging the whole np.arrays
                    linewidth=np.inf)  # one line for vectors


def myarray2string(array):
    return np.array2string(array, separator=', ').replace('\n', '')


class FakeLogger:
    def __init__(self):
        pass

    def _log_multiple(self, **kwargs):
        pass

    def _log(self, name, info, array):
        pass

    # AbstractDE
    def _AbstractDE_init(self, restart_eps_x, restart_eps_y, use_archive,
                        archive_size, population_size, p_best_rate,
                        variation_for_CR, scale_for_F):
        pass

    def _start_optimization(self, rng_seed):
        pass

    def _function_not_Describedfunction(self):
        pass

    def _improper_population_size(self, pop_size_given, pop_size_used):
        pass

    def _optimization_preparation(self, max_f_evals, dimension, number_of_best):
        pass

    def _start_generation(self, generations_done, generations_after_last_restart):
        pass

    def _end_optimization(self, generations_processed, best_member_ever, best_f_value_ever, restarts=None):
        pass

    def _restart(self):
        pass

    def _archive(self, archive):
        pass

    def _population(self, population, population_f_value):
        pass

    def _p_best(self, scores_index_sorted, current_worst_i, current_worst_f, current_p_best_i, current_p_best_f,
               current_best_i, current_best_f):
        pass

    def _update_solution(self, best_member_ever, best_f_value_ever):
        pass

    def _drawn_CR_F(self, drawn_M_CR, drawn_M_F, CR, F):
        pass

    def _members_for_mutation(self, r1, r2, x_r1, x_r2):
        pass

    def _population_trial(self, x_p_best, population_trial):
        pass

    def _p_best_draw(self, numbers_of_specimens_to_choose_from, p_best_members_indices):
        pass

    def _swap_population_trial(self, replace_with_trial_coord, population_trial):
        pass

    def _remove_from_archive(self, r):
        pass

    def _restarting_cond_x(self, numerator, denominator, restart_eps_x, abs=False):
        pass

    def _restarting_cond_y(self, numerator, denominator, restart_eps_y, abs=False):
        pass

    def _restarting_cond(self, type, numerator, denominator, restart_eps, abs=False):
        pass

    def _restarting_x(self, numerator, denominator, restart_eps_x):
        pass

    def _restarting_y(self, numerator, denominator, restart_eps_y):
        pass

    def _restarting(self, generations_after_last_restart, current_best_f):
        pass

    # SHADE
    def _SHADE_init(self, H, initial_M_CR, initial_M_F):
        pass

    def _unsuccessful_generation(self):
        pass

    def _indices_for_swap(self, f_difference, delta_f, indices_for_swap):
        pass

    def _new_CR_F(self, w, win_CR, mean_CR, win_F, mean_F):
        pass

    def _updated_CR_F(self, M_CR, M_F):
        pass

    def _attempts_of_back_to_domain(self, attempts):
        pass

    # DElo
    def _DElo_init(self, portion_of_top_players, player_elo_rating_rate,
                  task_elo_rating_rate, number_of_players):
        pass

    def _improper_player_amount(self, players_amount):
        pass

    def _elo_ratings(self, expected_results, actual_results, player_update,
                    players_rating, task_updates, task_ratings):
        pass

    def _top_players(self, top_players_indexes, top_players_indexes_r):
        pass

    def _indices_of_selected_players(self, indexes_of_selected_players):
        pass

    def _actual_results(self, actual_results):
        pass

    def _log_error(self, name, info):
        pass

    def _log_warning(self, name, info):
        pass

    def _turn_off(self):
        pass

    # DElo_Ties
    def _DElo_ties_init(self, history_for_ties, win_tie, tie_loss):
        pass

    # DElo_TQI
    def _joint_init(self, *args):
        pass

    def _joint_elo_ratings(self, *args):
        pass


class Logger:
    """Logs information on the process of optimizing with DE algorithm.

    An istance of it can be added to DE algorithm (DElo, DElo_ties_and_QI, SHADE) with the `logger` parameter.

    """
    def __init__(self, file='optimizer.log', what_to_log=None, optimizer_name='DE'):
        """Initialise the Logger

        Parameters
        ----------
        file: str
            The file to which the logged information will be provided.
        what_to_log: list, optional
            When provided, only the subset of the `what_to_log` list will be logged. Other information
            will not be logged.
            Useful when only some information is needed to speed up the process of logs gathering and
            make the log files take less place on hard drive.
        optimizer_name: string
            The name that will be logged on the beginning of the log with an "info" tag.
        """
        self.what_to_log = what_to_log

        self.pythonLogger = logging.getLogger(name="Optimizer_Logger" + file)
        self.pythonLogger.setLevel(logging.DEBUG)

        f_handler = logging.FileHandler(filename=file)
        f_handler.setLevel(logging.DEBUG)
        f_handler.setFormatter(logging.Formatter('%(levelname)s ; %(asctime)s ; %(logname)s ; %(message)s'))
        self.pythonLogger.addHandler(f_handler)

        self.array_variables = ['found solution', 'archive', 'population', 'population_f_value', 'scores_index_sorted',
                                'current_p_best_i', 'current_p_best_f', 'best_member_ever', 'drawn_M_CR', 'drawn_M_F',
                                'CR', 'F', 'indices_for_mutation', 'random_members_for_mutation', 'x_p_best',
                                'population_trial', 'numbers_of_specimens_to_choose_from', 'pbest_members_indices',
                                'swap', 'swapped_population_trial', 'remove_from_archive', 'f_difference', 'delta_f',
                                'indices_for_swap', 'w', 'win_CR', 'new_mean_M_CR', 'win_F', 'new_mean_M_F',
                                'updated_M_CR', 'updated_M_F', 'expected_results', 'actual_results', 'player_updates',
                                'player_ratings', 'task_updates', 'task_ratings', 'top_players_indices',
                                'indices_of_drawn_players_to_optimize', 'indices_of_selected_players']
        self.optimizer_name = optimizer_name

    def _log_multiple(self, **kwargs):
        for name, info in kwargs.items():
            self._log(name, info, name in self.array_variables)

    def _log(self, name, info, array=None):
        if self.what_to_log is not None and not name in self.what_to_log:
            return  # do not log anything
        if array is None:
            array = name in self.array_variables  # check if it is array and therefore has to be converted
        if array:
            info = myarray2string(info)  # convert the array

        extra = {'logname': name}
        self.pythonLoggerAdapter = logging.LoggerAdapter(self.pythonLogger, extra)
        self.pythonLoggerAdapter.debug(info, extra)

    # AbstractDE
    def _AbstractDE_init(self, restart_eps_x, restart_eps_y, use_archive,
                         archive_size, population_size, p_best_rate,
                         variation_for_CR, scale_for_F):
        self._log("info", "Optimizer initialised")
        self._log("restart_eps_x", restart_eps_x)
        self._log("restart_eps_y", restart_eps_y)
        self._log("use_archive", use_archive)
        self._log("archive_size", archive_size)
        self._log("population_size", population_size)
        self._log("p_best_rate", p_best_rate)
        self._log("variation_for_CR", variation_for_CR)
        self._log("scale_for_F", scale_for_F)

    def _start_optimization(self, rng_seed):
        self._log('info', 'Optimization started')
        self._log("rng_seed", rng_seed)

    def _function_not_Describedfunction(self):
        self._log_error('start', 'Optimization started with a function that is not of Describedfunction class')

    def _improper_population_size(self, pop_size_given, pop_size_used):
        self._log_warning('improper_population_size', f'pop_size_given={pop_size_given}, pop_size_used={pop_size_used}')

    def _optimization_preparation(self, max_f_evals, dimension, number_of_best):
        self._log('max_f_evals', max_f_evals)
        self._log('dimension', dimension)
        self._log('number_of_best', number_of_best)

    def _start_generation(self, generations_done, generations_after_last_restart):
        self._log('info', 'next generation started')
        self._log('generations_done', generations_done)
        self._log('generations_after_last_restart', generations_after_last_restart)

    def _end_optimization(self, generations_processed, best_member_ever, best_f_value_ever, restarts=None):
        self._log('info', f'Optimization ended')
        self._log('generations_processed', generations_processed)
        self._log('found_solution', best_member_ever, array=True)
        self._log('value_of_found_solution', best_f_value_ever)
        if restarts:  # for backward compatibility, when number of restarts was not logged
            self._log('restarts', restarts)

    def _restart(self):
        self._log('info', 'restart_search')

    def _archive(self, archive):
        self._log('archive', archive, array=True)

    def _population(self, population, population_f_value):
        self._log('population', population, array=True)
        self._log('population_f_value', population_f_value, array=True)

    def _p_best(self, scores_index_sorted, current_worst_i, current_worst_f, current_p_best_i, current_p_best_f,
                current_best_i, current_best_f):
        self._log('scores_index_sorted', scores_index_sorted, array=True)
        self._log('current_worst_i', current_worst_i)
        self._log('current_worst_f', current_worst_f)
        self._log('current_p_best_i', current_p_best_i, array=True)
        self._log('current_p_best_f', current_p_best_f, array=True)
        self._log('current_best_i', current_best_i)
        self._log('current_best_f', current_best_f)

    def _update_solution(self, best_member_ever, best_f_value_ever):
        self._log('info', 'best_member_ever and best_f_value_ever update')
        self._log('best_member_ever', best_member_ever, array=True)
        self._log('best_f_value_ever', best_f_value_ever)

    def _drawn_CR_F(self, drawn_M_CR, drawn_M_F, CR, F):
        self._log('drawn_M_CR', drawn_M_CR, array=True)
        self._log('drawn_M_F', drawn_M_F, array=True)
        self._log('CR', CR, array=True)
        self._log('F', F, array=True)

    def _members_for_mutation(self, r1, r2, x_r1, x_r2):
        self._log("indices_for_mutation", np.array((r1, r2)), array=True)
        self._log("random_members_for_mutation", np.array((x_r1, x_r2)), array=True)

    def _population_trial(self, x_p_best, population_trial):
        self._log("x_p_best", x_p_best, array=True)
        self._log("population_trial", population_trial, array=True)

    def _p_best_draw(self, numbers_of_specimens_to_choose_from, p_best_members_indices):
        self._log("numbers_of_specimens_to_choose_from", numbers_of_specimens_to_choose_from, array=True)
        self._log("pbest_members_indices", p_best_members_indices, array=True)

    def _swap_population_trial(self, replace_with_trial_coord, population_trial):
        self._log("swap", replace_with_trial_coord, array=True)
        self._log("swapped_population_trial", population_trial, array=True)

    def _remove_from_archive(self, r):
        self._log("remove_from_archive", r, array=True)

    def _restarting_cond_x(self, numerator, denominator, restart_eps_x, abs=False):
        self._restarting_cond("x", numerator, denominator, restart_eps_x, abs=abs)

    def _restarting_cond_y(self, numerator, denominator, restart_eps_y, abs=False):
        self._restarting_cond("y", numerator, denominator, restart_eps_y, abs=abs)

    def _restarting_cond(self, type, numerator, denominator, restart_eps, abs=False):
        self._log('restarting', type + ' restart condition met')
        if abs:
            denominator = 1
        if denominator != 0:
            self._log(f'{type}-restarting_value',
                     f'{numerator / denominator} which is smaller than restart_eps_{type}={restart_eps}')
        else:
            self._log(f'{type}-restarting_value',
                     f'0 which is smaller than restart_eps_{type}={restart_eps}')

    def _restarting(self, generations_after_last_restart, current_best_f):
        self._log('generations_after_last_restart_restarting', generations_after_last_restart)
        self._log('best_f_after_last_restart_restarting', current_best_f)

    def _attempts_of_back_to_domain(self, attempts):
        self._log('attempts_of_back_to_domain', attempts)

    # SHADE
    def _SHADE_init(self, H, initial_M_CR, initial_M_F):
        self._log('info', 'SHADE')
        self._log('history_size', H)
        self._log("initial_M_CR", initial_M_CR)
        self._log("initial_M_F", initial_M_F)

    def _unsuccessful_generation(self):
        self._log('info', 'this generation was unsuccessful')

    def _indices_for_swap(self, f_difference, delta_f, indices_for_swap):
        self._log('f_difference', f_difference, array=True)
        self._log('delta_f', delta_f, array=True)
        self._log('indices_for_swap', indices_for_swap, array=True)
        self._log('number_of_improvements_this_generation', sum(indices_for_swap))
        self._log('S', sum(delta_f))

    def _new_CR_F(self, w, win_CR, mean_CR, win_F, mean_F):
        self._log('w', w, array=True)
        self._log('win_CR', win_CR, array=True)
        self._log('new_mean_M_CR', mean_CR, array=True)
        self._log('win_F', win_F, array=True)
        self._log('new_mean_M_F', mean_F, array=True)

    def _updated_CR_F(self, M_CR, M_F):
        self._log('updated_M_CR', M_CR, array=True)
        self._log('updated_M_F', M_F, array=True)

    # DElo
    def _DElo_init(self, portion_of_top_players, player_elo_rating_rate,
                   task_elo_rating_rate, number_of_players):
        self._log("info", self.optimizer_name)
        self._log("portion_of_top_players", portion_of_top_players)
        self._log("player_elo_rating_rate", player_elo_rating_rate)
        self._log("task_elo_rating_rate", task_elo_rating_rate)
        self._log("number of players", number_of_players)

    def _improper_player_amount(self, players_amount):
        self._log_warning('players amount',
                         f'`players_amount` = {players_amount} is not a square of natural number.')

    def _elo_ratings(self, expected_results, actual_results, player_updates,
                     player_ratings, task_updates, task_ratings):
        self._log('expected_results', expected_results, array=True)
        self._log('actual_results', actual_results, array=True)
        self._log('player_updates', player_updates, array=True)
        self._log('player_ratings', player_ratings, array=True)
        self._log('task_updates', task_updates, array=True)
        self._log('task_ratings', task_ratings, array=True)

    def _top_players(self, top_players_indices, top_players_indices_r):
        self._log('top_players_indices', top_players_indices, array=True)
        self._log('indices_of_drawn_players_to_optimize', top_players_indices_r, array=True)

    def _indices_of_selected_players(self, indices_of_selected_players):
        self._log('indices_of_selected_players', indices_of_selected_players, array=True)

    # DElo_ties
    def _DElo_ties_init(self, history_for_ties, win_tie, tie_loss):
        self._log("history_for_ties_size", history_for_ties)
        self._log("win_tie_boundary", win_tie)
        self._log("tie_loss_boundary", tie_loss)

    # DElo_TQI
    def _joint_init(self, history_for_ties, win_tie, tie_loss, expectation_factor, player_elo_rating_rate_MOV,
                    task_elo_rating_rate_MOV):
        self._log("history_for_ties_size", history_for_ties)
        self._log("win_tie_boundary", win_tie)
        self._log("tie_loss_boundary", tie_loss)
        self._log('expectation_factor', expectation_factor, False)
        self._log('player_elo_rating_rate_MOV', player_elo_rating_rate_MOV, False)
        self._log('task_elo_rating_rate_MOV', task_elo_rating_rate_MOV, False)

    def _joint_elo_ratings(self, victory_odds, were_victorious, expected_relative_difference, actual_relative_difference,
                           player_updates, player_ratings, task_updates, task_ratings):
        self._log('victory_odds', victory_odds, array=True)
        self._log('were_victorious', were_victorious, array=True)
        self._log('expected_relative_difference', expected_relative_difference, array=True)
        self._log('actual_relative_difference', actual_relative_difference, array=True)
        self._log('player_updates', player_updates, array=True)
        self._log('player_ratings', player_ratings, array=True)
        self._log('task_updates', task_updates, array=True)
        self._log('task_ratings', task_ratings, array=True)

    def _log_error(self, name, info):
        extra = {'logname': name}
        self.pythonLoggerAdapter = logging.LoggerAdapter(self.pythonLogger, extra)
        self.pythonLoggerAdapter.error(info, extra)

    def _log_warning(self, name, info):
        extra = {'logname': name}
        self.pythonLoggerAdapter = logging.LoggerAdapter(self.pythonLogger, extra)
        self.pythonLoggerAdapter.warning(info)

    def _turn_off(self):
        for h in self.pythonLogger.handlers:
            self.pythonLogger.removeHandler(h)


class LogReader:
    """
    Read log created with *Logger*.

    Example
    --------
    >>> import delo
    >>> import numpy as np
    >>>
    >>> def square(x):
    ...     return np.sum(x ** 2, axis=1)
    >>>
    >>> file_name = 'square_opt.log'
    >>> logger = Logger(file=file_name)
    >>> described_function = delo.DescribedFunction(square, dimension=2, domain_lower_limit=-10, domain_upper_limit=10)
    >>> algorithm = delo.DElo(10, logger=logger)
    >>>
    >>> algorithm.optimize(described_function, rng_seed=2022)
    >>>
    >>> logreader = LogReader(file_name)
    >>> best_fs = logreader.read_variable('current_best_f')
    Looking for current_best_f in log file
    Found 100 occurences of `current_best_f`.
    >>> print(best_fs[:5])
    [27.935020304146946, 13.606498015936902, 4.37874090480261, 2.9852266609374456, 0.29795569609533]
    """

    def __init__(self, file):
        """
        Constructor

        Parameters
        ----------
        file : str
            path to \*.log file created with ``LogReader``.
        """
        self.file = file
        self._variable_types = {'info': 'str',
                                'restart_eps_x': 'float',
                                'restart_eps_y': 'float',
                                'rng_seed': 'int',
                                'use_archive': 'bool',
                                'archive_size': 'int',
                                'population_size': 'int',
                                'p_best_rate': 'float',
                                'portion_of_top_players': 'float',
                                'player_elo_rating_rate': 'float',
                                'task_elo_rating_rate': 'float',
                                'number of players': 'int',
                                'max_f_evals': 'int',
                                'dimension': 'int',
                                'number_of_best': 'int',
                                'archive': 'np.array',
                                'population': 'np.array',
                                'population_f_value': 'np.array',
                                'scores_index_sorted': 'np.array',
                                'current_worst_i': 'int',
                                'current_worst_f': 'float',
                                'current_p_best_i': 'int',
                                'current_p_best_f': 'float',
                                'current_best_i': 'int',
                                'current_best_f': 'float',
                                'best_member_ever': 'np.array',
                                'best_f_value_ever': 'float',
                                'generations_done': 'int',
                                'generations_after_last_restart': 'int',
                                'top_players_indexes': 'np.array',
                                'top_players_indices': 'np.array',
                                'indexes of drawn players to optimize': 'np.array',
                                'indices of drawn players to optimize': 'np.array',
                                'drawn_M_CR': 'np.array',
                                'drawn_M_F': 'np.array',
                                'CR': 'np.array',
                                'F': 'np.array',
                                'p': 'np.array',
                                'indexes_of_selected_players': 'np.array',
                                'indices_of_selected_players': 'np.array',
                                'indices_for_mutation': 'np.array',
                                'random_members_for_mutation': 'np.array',
                                'numbers_of_specimens_to_choose_from': 'np.array',
                                'pbest_members_indices': 'np.array',
                                'pbest_members_indexes': 'np.array',
                                'x_pbest': 'np.array',
                                'x_p_best': 'np.array',
                                'population_trial': 'np.array',
                                'swapped_population_trial': 'np.array',
                                'swap': 'np.array',
                                'f_difference': 'np.array',
                                'delta_f': 'np.array',
                                'indices_for_swap': 'np.array',
                                'number of improvements this generation': 'int',
                                'expected_results': 'np.array',
                                'actual_results': 'np.array',
                                'player_update': 'np.array',
                                'player_updates': 'np.array',
                                'players.rating': 'np.array',
                                'player_ratings': 'np.array',
                                'task_updates': 'np.array',
                                'task_ratings': 'np.array',
                                'where_use_archive': 'np.array',
                                'remove_from_archive': 'np.array',
                                'restarting': 'str',
                                'x-restarting_value': 'float',
                                'y-restarting_value': 'float',
                                'victory_odds': 'np.array',
                                'were_victorious': 'np.array',
                                'expected_relative_difference': 'np.array',
                                'actual_relative_difference': 'np.array',
                                'history_for_ties_size': 'int',
                                'win_tie_boundary': 'float',
                                'tie_loss_boundary': 'float',
                                'restarts': 'int',
                                'variation_for_CR': 'float',
                                'scale_for_F': 'float',
                                'generations_after_last_restart_restarting': 'int',
                                'best_f_after_last_restart_restarting': 'float',
                                'value_of_found_solution': 'float',
                                'generations_processed': 'int',
                                'attempts_of_back_to_domain': 'int'}

    def read_variables(self, variable_name_list, type_name_list=None, silence=False):
        """
        Get a list of multiple variables recorded in log.

        Parameters
        ----------
        variable_name_list : list of str
            names of variables to be read from log.
        type_name_list : list of str, optional
            types of variables to be read. If not provided, default variable types will be used.
            Acceptable types: "str", "int", "float", "bool", "list", "np.array".
        silence : bool
            Whether print messages about progess (False) ot not (True).

        Returns
        -------
        dict
            keys: variable names, values: lists of variable values in consecutive iterations.
        """

        outcome = dict()
        for i in range(len(variable_name_list)):
            try:
                if type_name_list is None:
                    local_outcome = self.read_variable(variable_name_list[i], type_name_list, silence=silence)
                else:
                    local_outcome = self.read_variable(variable_name_list[i], type_name_list[i], silence=silence)
                outcome[variable_name_list[i]] = local_outcome
            except:
                print(f'An exception occurred by {variable_name_list[i]}. Proceeding further.')
        return outcome

    def read_variable(self, variable_name, type_name=None, silence=False):
        """
        Get a list of a single variable recorded in logs.

        Parameters
        ----------
        variable_name : str
            name of variable to be read from log. Run ``get_variable_names`` method for acceptable values.
        type_name : str, optional
            type of variable to be read. If not provided, default variable type will be used.
            Acceptable types: "str", "int", "float", "bool", "list", "np.array".
        silence : bool
            Whether print messages about progess (False) ot not (True).

        Returns
        -------
        list
            variable values in consecutive iterations.
        """

        if not silence:
            print(f"Looking for {variable_name} in log file")
        if type_name is None:
            if variable_name not in self._variable_types.keys():
                raise Exception(f"`{variable_name}` has no type preset. Enter a type or edit LogReader class")
            type_name = self._variable_types[variable_name]
        else:
            if variable_name in self._variable_types.keys():
                if type_name in ['np.ndarray', 'np.array', 'array']:
                    type_name = 'np.array'
                if self._variable_types[variable_name] != type_name:
                    warnings.warn(
                        f"Supported `type_name` is {type_name}, but preset type_name is {self._variable_types[variable_name]}")

        # Structure of lines in file:
        # DEBUG ; 2021-11-14 20:52:34,979 ; restart_eps_x ; None
        # Loglevel ; time ; name ; value

        regex = "^.*\s;\s[0-9]{4}-[0-9]{2}-[0-9]{2}\s[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\s;\s" + variable_name + "\s;\s.*$"
        outcome = []
        with open(self.file, 'r') as file:
            for line in file:
                if re.match(regex, line) is None:
                    continue
                try:
                    processed = self._process_line(line, type_name)
                    outcome.append(processed['variable_value'])
                except:
                    print(f"Error in parsing a value of `{variable_name}` to {type_name}. Proceeding further.")

        if not silence:
            print(f"Found {len(outcome)} occurences of `{variable_name}`.")
        return outcome

    def get_variable_names(self):
        """
        Get a list of names of variables recorded in logs.

        Returns
        -------
        list of str
            names of variables stored in logs.
        """
        variable_names = []
        regex = "^.*\s;\s[0-9]{4}-[0-9]{2}-[0-9]{2}\s[0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\s;\s.*\s;\s.*$"
        with open(self.file, 'r') as file:
            for line in file:
                if re.match(regex, line) is None:
                    print(f"Found wrong line!")
                    continue
                splitted = line.split(' ; ')
                new_variable_name = splitted[2]
                if new_variable_name not in variable_names:
                    variable_names.append(new_variable_name)
        print(f"Found {len(variable_names)} different variables.")
        return variable_names

    def _process_line(self, line, type_name):
        splitted = line.split(' ; ')
        if len(splitted) != 4:
            raise Exception('Line not processed')
        return {'levelname': splitted[0],
                'asctime': time.strptime(splitted[1], "%Y-%m-%d %H:%M:%S,%f"),
                'variable_name': splitted[2],
                'variable_value': self._parse_value(splitted[3], type_name)}

    def _parse_value(self, value, type_name):
        value = value[:-1]  # last character is '\n'
        if value == 'None':
            return None
        elif type_name == 'str':
            return value
        elif type_name in ['int', 'float', 'bool', 'list']:
            return ast.literal_eval(value)
        elif type_name in ['np.ndarray', 'np.array', 'array']:
            return np.asarray(ast.literal_eval(value))
        else:
            raise Exception('Format not supported')

    def read_solver_configuration(self):
        """
        Print on console initial parameters and hyperparameters.
        """
        with open(self.file, 'r') as file:
            for line in file:
                if re.search("info ; restart_search", line) is not None:  # this is end of configuration
                    break
                print(line)


class FakePrinter:  # fake printer will not print anything
    def __init__(self, print_every=100):
        pass

    def start_optimization(self, generations_processed, budget):
        pass

    def generation(self, generations_done, generations_after_last_restart, current_best_f,
                   best_f_value_ever, number_of_improvements):
        return False

    def restarting(self, generations_after_last_restart, current_best_f):
        pass

    def optimizing_complete(self, restarts, generations_done, remaining_evals, generations_processed,
                            best_f_value_ever):
        pass


class Printer:
    def __init__(self, print_every=100):
        self.print_every = int(print_every)

    def start_optimization(self, budget, seed):
        print(f"Optimizing with budget of {budget} and seed = {seed}:\n")

    def generation(self, generations_done, generations_after_last_restart, current_best_f,
                   best_f_value_ever, number_of_improvements):
        if generations_done % self.print_every == 0:
            print("{generations_done}({generations_after_last_restart}). f(current_best) = {current_best_f:.4f}; "
                  "f(best_ever) = {best_f_value_ever:.4f}; improvements since last print = {number_of_improvements}".format(
                generations_done=generations_done,
                generations_after_last_restart=generations_after_last_restart,
                current_best_f=current_best_f,
                best_f_value_ever=best_f_value_ever,
                number_of_improvements=number_of_improvements))
            return True
        return False

    def restarting(self, generations_after_last_restart, current_best_f):
        print("Restarting after {generations} generations since last restart;\n"
              "f(current_best) = {current_best_f:.4f}\n".format(
            generations=generations_after_last_restart, current_best_f=current_best_f))

    def optimizing_complete(self, restarts, generations_done, remaining_evals, generations_processed,
                            best_f_value_ever):
        print(f"\nOptimizing complete after {restarts} restarts, {generations_done} generations and"
              f"with {remaining_evals} remaining evaluations")
        print("{}. f(best_ever) = {best_f_value_ever:.4f}".format(
            generations_processed, best_f_value_ever=best_f_value_ever))
