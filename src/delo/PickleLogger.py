from .Logger import *
import os

class PickleLogger(Logger):
    def __init__(self, file='optimizer.log', what_to_log=None, optimizer_name='DE'):
        super().__init__(file=file, what_to_log=what_to_log, optimizer_name=optimizer_name)
        absolute_path = os.path.relpath(file)
        dir_name_candidate, ext = os.path.splitext(absolute_path)
        dir_name_candidate += '_objects'
        if os.path.exists(dir_name_candidate):
            dir_name_suffix=0
            while os.path.exists(dir_name_candidate+str(dir_name_suffix)):
                dir_name_suffix += 1
            dir_name_candidate=dir_name_candidate+str(dir_name_suffix)
        self.dir_for_arrays=dir_name_candidate
        os.mkdir(self.dir_for_arrays)
        self.generation_count=0
        self.batch_dict={}

    def log(self, name, info, array=None):
        if self.what_to_log is not None and not name in self.what_to_log:
            return
        if array is None:
            array = name in self.array_variables
        if array:
            self.batch_dict[name]=info
            info = self.get_filepath_for_nparray(rel_to_logfile=True)
        super().log(name, info, array=False)

    def start_generation(self, generations_done, generations_after_last_restart):
        self.log_batch()
        super().start_generation(generations_done, generations_after_last_restart)

    def restarting(self, generations_after_last_restart, current_best_f):
        self.log_batch()
        super().restarting(generations_after_last_restart, current_best_f)

    def log_batch(self):
        filepath = self.get_filepath_for_nparray()
        np.savez_compressed(filepath, **self.batch_dict)
        self.batch_dict={}
        self.generation_count += 1

    def get_filepath_for_nparray(self, rel_to_logfile=False):
        filename = 'gen' + str(self.generation_count) + '.npz'
        if rel_to_logfile:
            dirname = os.path.basename(self.dir_for_arrays)
        else:
            dirname = self.dir_for_arrays
        filepath = os.path.join(dirname, filename)
        return filepath


class PickleLogReader(LogReader):
    def process_line(self, line, type_name):
        splitted = line.split(' ; ')
        if len(splitted) != 4:
            raise Exception('Line not processed')
        variable_name = splitted[2]
        raw_value = splitted[3]
        if type_name == 'np.array' and os.path.isfile(raw_value[:-1]):
            with np.load(raw_value[:-1]) as data:
                parsed_value = data[variable_name]
        else:
            parsed_value = self.parse_value(raw_value, type_name)
        return {'levelname': splitted[0],
                'asctime': time.strptime(splitted[1], "%Y-%m-%d %H:%M:%S,%f"),
                'variable_name': variable_name,
                'variable_value': parsed_value}


