class FunctionNoCollableException(Exception):
    def __init__(self, message="Funciton provided is not callable"):
        self.message = message
        super().__init__(self.message)

class VariableNotIntException(Exception):
    def __init__(self, message="Variable should be int"):
        self.message = message
        super().__init__(self.message)

class ImproperIntException(Exception):
    def __init__(self, message="Provided integer is improper"):
        self.message = message
        super().__init__(self.message)

class ImproperDomainLimitsException(Exception):
    def __init__(self, message="Provided domain limits are improper"):
        self.message = message
        super().__init__(self.message)

class ImproperRestartEpsilonException(Exception):
    def __init__(self, message="Provided restart epsilon is improper"):
        self.message = message
        super().__init__(self.message)

class NegativeArchiveException(Exception):
    def __init__(self, message="Provided maximum size for archive is negative"):
        self.message = message
        super().__init__(self.message)

class NonIntArchiveException(Exception):
    def __init__(self, message="Provided maximum size for archive is not int"):
        self.message = message
        super().__init__(self.message)

class NonPositivePopulationSizeException(Exception):
    def __init__(self, message="Provided size for population is not positive number"):
        self.message = message
        super().__init__(self.message)

class PopulationSize1Exception(Exception):
    def __init__(self, message="Provided size for population is equal to 1"):
        self.message = message
        super().__init__(self.message)

class PopulationSizeNotIntException(Exception):
    def __init__(self, message="Provided size for population is not integer"):
        self.message = message
        super().__init__(self.message)

class p_best_rateOutOf01Exception(Exception):
    def __init__(self, message="Provided p_best_rate is out of [0,1] range"):
        self.message = message
        super().__init__(self.message)

class ChoppedCauchyImproperLocationException(Exception):
    def __init__(self, message='Improper size and number of numbers to generate in chopped_cauchy()'):
        self.message = message
        super().__init__(self.message)