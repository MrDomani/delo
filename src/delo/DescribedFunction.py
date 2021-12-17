import numpy as np
from .CustomExceptions import *

class DescribedFunction:
    """
    Function with attributes

    Wrapper for function objects, that contains info about function's domain.

    Attributes
    ----------
    dimension : int
        Dimension of function's domain.
    domain_lower_limit : np.array of `dimension` size
        The lower limits of function domain.
    domain_upper_limit : np.array of `dimension` size
        The upper limit of function domain.

    Methods
    -------
    call
        Call the passed in constructor.

    Examples
    --------
    >>> described_arcsin = delo.DescribedFunction(np.argsin, dimension=1,
    ...                    domain_lower_limit=-1, domain_upper_limit=1)

    DescribedFunction was created to be used in DE algorithms. Specifying domain limits is required
    >>> described_square_root = delo.DescribedFunction(np.sqrt, dimension=1,
    ...                         domain_lower_limit=0, domain_upper_limit=5)
    """
    def __init__(self, function, dimension, domain_lower_limit=None, domain_upper_limit=None, name=None):
        """
        Function with attributes

        Parameters
        ----------
        function : callable
            function with 2D `np.ndarray` input and scalar output.
        dimension : int
            Dimension of function's domain.
        domain_lower_limit : np.array, float, optional
            the lower limit of function domain. If None, it will be set to be repeated -10.
        domain_upper_limit : np.array, float, optional
            is the upper limit of function domain. If None, it will be set to be repeated 10.
        """
        if not callable(function):
            raise FunctionNoCollableException()

        self.call = function

        if type(dimension) != int:
            raise VariableNotIntException(f"dimention should be an int, but is {dimension.__class__.__name__}")
        if dimension < 1:
            raise ImproperIntException(f"Provided dimension = {dimension} is smaller than 1")
        self.dimension = dimension

        self.domain_lower_limit = self.process_domain_limit(domain_lower_limit, lower=True)
        self.domain_upper_limit = self.process_domain_limit(domain_upper_limit)

        if any(self.domain_upper_limit - self.domain_lower_limit < 0):
            raise ImproperDomainLimitsException()

        if any(self.domain_upper_limit - self.domain_lower_limit == 0):
            raise ImproperDomainLimitsException("One or more dimensions have equal lower and upper limits."
                             "In case it is wanted, this behaviour has to be implemented as funciton modificaiton.")


        self.name = name

    def process_domain_limit(self, limit, lower=False):
        if limit is None:
            out = 10 * np.ones(self.dimension)
            if lower:
                out = -out
            return out
        if isinstance(limit, float) or isinstance(limit, int):
            return limit * np.ones(self.dimension)
        elif isinstance(limit, np.ndarray) and limit.shape == (self.dimension,):
            return limit
        else:
            raise Exception('If provided, domain limits must be an int, float or np.ndarray of shape (`dimension`,).')
