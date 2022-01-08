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

    def log_multiple(self, **kwargs):
        pass

    def log(self, name, info, array):
        pass

    # AbstractDE
    def AbstractDE_init(self, restart_eps_x, restart_eps_y, use_archive,
                        archive_size, population_size, p_best_rate,
                        variation_for_CR, scale_for_F):
        pass

    def start_optimization(self, rng_seed):
        pass

    def function_not_Describedfunction(self):
        pass

    def improper_population_size(self, pop_size_given, pop_size_used):
        pass

    def optimization_preparation(self, max_f_evals, dimension, number_of_best):
        pass

    def start_generation(self, generations_done, generations_after_last_restart):
        pass

    def end_optimization(self, generations_processed, best_member_ever, best_f_value_ever, restarts=None):
        pass

    def restart(self):
        pass

    def archive(self, archive):
        pass

    def population(self, population, population_f_value):
        pass

    def p_best(self, scores_index_sorted, current_worst_i, current_worst_f, current_p_best_i, current_p_best_f,
               current_best_i, current_best_f):
        pass

    def update_solution(self, best_member_ever, best_f_value_ever):
        pass

    def drawn_CR_F(self, drawn_M_CR, drawn_M_F, CR, F):
        pass

    def members_for_mutation(self, r1, r2, x_r1, x_r2):
        pass

    def population_trial(self, x_p_best, population_trial):
        pass

    def p_best_draw(self, numbers_of_specimens_to_choose_from, p_best_members_indices):
        pass

    def swap_population_trial(self, replace_with_trial_coord, population_trial):
        pass

    def remove_from_archive(self, r):
        pass

    def restarting_cond_x(self, numerator, denominator, restart_eps_x, abs=False):
        pass

    def restarting_cond_y(self, numerator, denominator, restart_eps_y, abs=False):
        pass

    def restarting_cond(self, type, numerator, denominator, restart_eps, abs=False):
        pass

    def restarting_x(self, numerator, denominator, restart_eps_x):
        pass

    def restarting_y(self, numerator, denominator, restart_eps_y):
        pass

    def restarting(self, generations_after_last_restart, current_best_f):
        pass

    # SHADE
    def SHADE_init(self, H, initial_M_CR, initial_M_F):
        pass

    def unsuccessful_generation(self):
        pass

    def indices_for_swap(self, f_difference, delta_f, indices_for_swap):
        pass

    def new_CR_F(self, w, win_CR, mean_CR, win_F, mean_F):
        pass

    def updated_CR_F(self, M_CR, M_F):
        pass

    def attempts_of_back_to_domain(self, attempts):
        pass

    # DElo
    def DElo_init(self, portion_of_top_players, player_elo_rating_rate,
                  task_elo_rating_rate, number_of_players):
        pass

    def improper_player_amount(self, players_amount):
        pass

    def elo_ratings(self, expected_results, actual_results, player_update,
                    players_rating, task_updates, task_ratings):
        pass

    def top_players(self, top_players_indexes, top_players_indexes_r):
        pass

    def indices_of_selected_players(self, indexes_of_selected_players):
        pass

    def actual_results(self, actual_results):
        pass

    def log_error(self, name, info):
        pass

    def log_warning(self, name, info):
        pass

    def turn_off(self):
        pass

    # DElo_Ties
    def DElo_ties_init(self, history_for_ties, win_tie, tie_loss):
        pass

    # DElo_TQI
    def joint_init(self, *args):
        pass

    def joint_elo_ratings(self, *args):
        pass


class Logger:
    def __init__(self, file='optimizer.log', what_to_log=None, optimizer_name='DE'):
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

    def log_multiple(self, **kwargs):
        for name, info in kwargs.items():
            self.log(name, info, name in self.array_variables)

    def log(self, name, info, array=None):
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
    def AbstractDE_init(self, restart_eps_x, restart_eps_y, use_archive,
                        archive_size, population_size, p_best_rate,
                        variation_for_CR, scale_for_F):
        self.log("info", "Optimizer initialised")
        self.log("restart_eps_x", restart_eps_x)
        self.log("restart_eps_y", restart_eps_y)
        self.log("use_archive", use_archive)
        self.log("archive_size", archive_size)
        self.log("population_size", population_size)
        self.log("p_best_rate", p_best_rate)
        self.log("variation_for_CR", variation_for_CR)
        self.log("scale_for_F", scale_for_F)

    def start_optimization(self, rng_seed):
        self.log('info', 'Optimization started')
        self.log("rng_seed", rng_seed)

    def function_not_Describedfunction(self):
        self.log_error('start', 'Optimization started with a function that is not of Describedfunction class')

    def improper_population_size(self, pop_size_given, pop_size_used):
        self.log_warning('improper_population_size', f'pop_size_given={pop_size_given}, pop_size_used={pop_size_used}')

    def optimization_preparation(self, max_f_evals, dimension, number_of_best):
        self.log('max_f_evals', max_f_evals)
        self.log('dimension', dimension)
        self.log('number_of_best', number_of_best)

    def start_generation(self, generations_done, generations_after_last_restart):
        self.log('info', 'next generation started')
        self.log('generations_done', generations_done)
        self.log('generations_after_last_restart', generations_after_last_restart)

    def end_optimization(self, generations_processed, best_member_ever, best_f_value_ever, restarts=None):
        self.log('info', f'Optimization ended')
        self.log('generations_processed', generations_processed)
        self.log('found_solution', best_member_ever, array=True)
        self.log('value_of_found_solution', best_f_value_ever)
        if restarts:  # for backward compatibility, when number of restarts was not logged
            self.log('restarts', restarts)

    def restart(self):
        self.log('info', 'restart_search')

    def archive(self, archive):
        self.log('archive', archive, array=True)

    def population(self, population, population_f_value):
        self.log('population', population, array=True)
        self.log('population_f_value', population_f_value, array=True)

    def p_best(self, scores_index_sorted, current_worst_i, current_worst_f, current_p_best_i, current_p_best_f,
               current_best_i, current_best_f):
        self.log('scores_index_sorted', scores_index_sorted, array=True)
        self.log('current_worst_i', current_worst_i)
        self.log('current_worst_f', current_worst_f)
        self.log('current_p_best_i', current_p_best_i, array=True)
        self.log('current_p_best_f', current_p_best_f, array=True)
        self.log('current_best_i', current_best_i)
        self.log('current_best_f', current_best_f)

    def update_solution(self, best_member_ever, best_f_value_ever):
        self.log('info', 'best_member_ever and best_f_value_ever update')
        self.log('best_member_ever', best_member_ever, array=True)
        self.log('best_f_value_ever', best_f_value_ever)

    def drawn_CR_F(self, drawn_M_CR, drawn_M_F, CR, F):
        self.log('drawn_M_CR', drawn_M_CR, array=True)
        self.log('drawn_M_F', drawn_M_F, array=True)
        self.log('CR', CR, array=True)
        self.log('F', F, array=True)

    def members_for_mutation(self, r1, r2, x_r1, x_r2):
        self.log("indices_for_mutation", np.array((r1, r2)), array=True)
        self.log("random_members_for_mutation", np.array((x_r1, x_r2)), array=True)

    def population_trial(self, x_p_best, population_trial):
        self.log("x_p_best", x_p_best, array=True)
        self.log("population_trial", population_trial, array=True)

    def p_best_draw(self, numbers_of_specimens_to_choose_from, p_best_members_indices):
        self.log("numbers_of_specimens_to_choose_from", numbers_of_specimens_to_choose_from, array=True)
        self.log("pbest_members_indices", p_best_members_indices, array=True)

    def swap_population_trial(self, replace_with_trial_coord, population_trial):
        self.log("swap", replace_with_trial_coord, array=True)
        self.log("swapped_population_trial", population_trial, array=True)

    def remove_from_archive(self, r):
        self.log("remove_from_archive", r, array=True)

    def restarting_cond_x(self, numerator, denominator, restart_eps_x, abs=False):
        self.restarting_cond("x", numerator, denominator, restart_eps_x, abs=abs)

    def restarting_cond_y(self, numerator, denominator, restart_eps_y, abs=False):
        self.restarting_cond("y", numerator, denominator, restart_eps_y, abs=abs)

    def restarting_cond(self, type, numerator, denominator, restart_eps, abs=False):
        self.log('restarting', type + ' restart condition met')
        if abs:
            denominator = 1
        if denominator != 0:
            self.log(f'{type}-restarting_value',
                     f'{numerator / denominator} which is smaller than restart_eps_{type}={restart_eps}')
        else:
            self.log(f'{type}-restarting_value',
                     f'0 which is smaller than restart_eps_{type}={restart_eps}')

    def restarting(self, generations_after_last_restart, current_best_f):
        self.log('generations_after_last_restart_restarting', generations_after_last_restart)
        self.log('best_f_after_last_restart_restarting', current_best_f)

    def attempts_of_back_to_domain(self, attempts):
        self.log('attempts_of_back_to_domain', attempts)

    # SHADE
    def SHADE_init(self, H, initial_M_CR, initial_M_F):
        self.log('info', 'SHADE')
        self.log('history_size', H)
        self.log("initial_M_CR", initial_M_CR)
        self.log("initial_M_F", initial_M_F)

    def unsuccessful_generation(self):
        self.log('info', 'this generation was unsuccessful')

    def indices_for_swap(self, f_difference, delta_f, indices_for_swap):
        self.log('f_difference', f_difference, array=True)
        self.log('delta_f', delta_f, array=True)
        self.log('indices_for_swap', indices_for_swap, array=True)
        self.log('number_of_improvements_this_generation', sum(indices_for_swap))
        self.log('S', sum(delta_f))

    def new_CR_F(self, w, win_CR, mean_CR, win_F, mean_F):
        self.log('w', w, array=True)
        self.log('win_CR', win_CR, array=True)
        self.log('new_mean_M_CR', mean_CR, array=True)
        self.log('win_F', win_F, array=True)
        self.log('new_mean_M_F', mean_F, array=True)

    def updated_CR_F(self, M_CR, M_F):
        self.log('updated_M_CR', M_CR, array=True)
        self.log('updated_M_F', M_F, array=True)

    # DElo
    def DElo_init(self, portion_of_top_players, player_elo_rating_rate,
                  task_elo_rating_rate, number_of_players):
        self.log("info", self.optimizer_name)
        self.log("portion_of_top_players", portion_of_top_players)
        self.log("player_elo_rating_rate", player_elo_rating_rate)
        self.log("task_elo_rating_rate", task_elo_rating_rate)
        self.log("number of players", number_of_players)

    def improper_player_amount(self, players_amount):
        self.log_warning('players amount',
                         f'`players_amount` = {players_amount} is not a square of natural number.')

    def elo_ratings(self, expected_results, actual_results, player_updates,
                    player_ratings, task_updates, task_ratings):
        self.log('expected_results', expected_results, array=True)
        self.log('actual_results', actual_results, array=True)
        self.log('player_updates', player_updates, array=True)
        self.log('player_ratings', player_ratings, array=True)
        self.log('task_updates', task_updates, array=True)
        self.log('task_ratings', task_ratings, array=True)

    def top_players(self, top_players_indices, top_players_indices_r):
        self.log('top_players_indices', top_players_indices, array=True)
        self.log('indices_of_drawn_players_to_optimize', top_players_indices_r, array=True)

    def indices_of_selected_players(self, indices_of_selected_players):
        self.log('indices_of_selected_players', indices_of_selected_players, array=True)

    # DElo_ties
    def DElo_ties_init(self, history_for_ties, win_tie, tie_loss):
        self.log("history_for_ties_size", history_for_ties)
        self.log("win_tie_boundary", win_tie)
        self.log("tie_loss_boundary", tie_loss)

    # DElo_TQI
    def joint_init(self, history_for_ties, win_tie, tie_loss, expectation_factor, player_elo_rating_rate_MOV,
                   task_elo_rating_rate_MOV):
        self.log("history_for_ties_size", history_for_ties)
        self.log("win_tie_boundary", win_tie)
        self.log("tie_loss_boundary", tie_loss)
        self.log('expectation_factor', expectation_factor, False)
        self.log('player_elo_rating_rate_MOV', player_elo_rating_rate_MOV, False)
        self.log('task_elo_rating_rate_MOV', task_elo_rating_rate_MOV, False)

    def joint_elo_ratings(self, victory_odds, were_victorious, expected_relative_difference, actual_relative_difference,
                          player_updates, player_ratings, task_updates, task_ratings):
        self.log('victory_odds', victory_odds, array=True)
        self.log('were_victorious', were_victorious, array=True)
        self.log('expected_relative_difference', expected_relative_difference, array=True)
        self.log('actual_relative_difference', actual_relative_difference, array=True)
        self.log('player_updates', player_updates, array=True)
        self.log('player_ratings', player_ratings, array=True)
        self.log('task_updates', task_updates, array=True)
        self.log('task_ratings', task_ratings, array=True)

    def log_error(self, name, info):
        extra = {'logname': name}
        self.pythonLoggerAdapter = logging.LoggerAdapter(self.pythonLogger, extra)
        self.pythonLoggerAdapter.error(info, extra)

    def log_warning(self, name, info):
        extra = {'logname': name}
        self.pythonLoggerAdapter = logging.LoggerAdapter(self.pythonLogger, extra)
        self.pythonLoggerAdapter.warning(info)

    def turn_off(self):
        for h in self.pythonLogger.handlers:
            self.pythonLogger.removeHandler(h)


class LogReader:
    """
    Read log created with `Logger`.

    Example
    --------
    >>> def square(x):
    ...     return x ** 2
    >>> file_name = 'square_opt.log'
    >>> logger = Logger(file=file_name)
    >>> described_function = delo.DescribedFunction(square, dimension=2, domain_lower_limit=-10, domain_upper_limit=10)
    >>> algorithm = delo.DElo(10, logger=logger)
    >>> algorithm.optimize(described_function, rng_seed=2022)
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
            path to *.log file created with `LogReader`.
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
            name of variable to be read from log. Run `get_variable_names` for acceptable values.
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
