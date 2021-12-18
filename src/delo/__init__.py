"""
DElo

Some more exact documantation about DElo package
"""

from .DElo import DElo
from .SHADE import SHADE
from .DescribedFunction import DescribedFunction
from .Logger import *
from .PickleLogger import *
from .DElo_ties import DElo_ties
from .DElo_ties_and_QI import DElo_ties_and_QI
from .QualityOfImprovement import DElo_QualityOfImprovement

__all__ = ['DElo', 'SHADE', 'DescribedFunction', 'Logger', 'LogReader',
           'PickleLogger', 'PickleLogReader',
           'DElo_ties', 'DElo_ties_and_QI', 'DElo_QualityOfImprovement']

