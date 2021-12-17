import numpy as np
from .CustomExceptions import ChoppedCauchyImproperLocationException


MAX_DEPTH_OF_RECURSION = 500


def chopped_normal(rng, size, location=np.array([0.5]), variation=0.1):
    """Generate values from normal distribution trimmed to [0,1]"""
    x = np.sqrt(variation) * rng.standard_normal(size) + location
    x[x < 0] = 0
    x[x > 1] = 1
    return x


def chopped_cauchy(rng, size, location=np.array([0.5]), scale=0.1, number_of_iterations=0):
    """
    Generate values from modified cauchy distribution.
    Every location can be different. Every scale has to be the same
    Modification is that if generated number is greater than 1, it is set to 1.
                         if generated number is less than 0, it is regenerated.
    """
    if np.all(location == 0) and scale == 0:
        return np.ones(size) * 0.00001  # this is the special case scale == 0, which is not recommended
    if len(location) == 1 and size != 1:
        location = location[0]  # it is float now
    if type(location) in [np.float64, float]:
        location = location * np.ones(size)
    if size != len(location):
        raise ChoppedCauchyImproperLocationException(
            f'Improper size and number of numbers to generate in chopped_cauchy().\n' +
            f'{size} numbers expected, but {len(location)} locations were given.')
    if number_of_iterations > MAX_DEPTH_OF_RECURSION:
        raise Exception('Did not managed to draw a proper F')  # this had never happened

    x = rng.standard_cauchy(size=size) * scale
    x += location  # Change of location based on https://en.wikipedia.org/wiki/Cauchy_distribution#Transformation_properties

    x[x > 1] = 1
    n1 = sum(x <= 0)  # number of missed trials
    if n1 > 0:
        x[x <= 0] = chopped_cauchy(rng=rng, size=n1, location=location[x <= 0], scale=scale,
                                   number_of_iterations=number_of_iterations + 1)  # regeneration of missed trials

    return x

def choose_2_columns_of_integers(rng, nrow=2, matrix_of_restrictions=np.array([[0, 0], [10, 15]])):
    """
    Choose integers with restrictions

    Draw integers from uniform distribution. It will make sure in every row there is no 2 the same integers and
    that in i-th row there is no integer i. The integers are drawn from restrictions
    provided in matrix_of_restrictions parameter.

    Parameters
    ----------
    rng : np.random._generator.Generator
        numpy random generator that will be used to draw numbers
    nrow : positive int
        number of integers to be drawn for both columns
    matrix of restrictions : np.ndarray of shape (2,2)
        The [0,0] element is lower restriction for the first column,
        the [0,1] element is lower restriction for the second column,
        the [1,0] element is upper restriction for the first column,
        the [1,1] element is upper restriction for the second column.

    Returns
    -------
    2D np.ndarray of (`nrow`, 2) shape
        In every row there is no 2 the same integers and
    that in i-th row there is no integer i.
    """
    indices_1 = rng.integers(matrix_of_restrictions[0,0], matrix_of_restrictions[1,0], size=nrow)
    indices_2 = rng.integers(matrix_of_restrictions[0,1], matrix_of_restrictions[1,1], size=nrow)

    number_of_iterations = 0
    # In each pair indices have to be:
    # 1) different from each other
    # 2) different from i - row number
    while True:
        indices_drawn_wrong = np.logical_or(indices_1 == indices_2,
                                          indices_1 == np.arange(nrow),
                                          indices_2 == np.arange(nrow))
        if not any(indices_drawn_wrong):  # all are correct
            break
        number_of_iterations += 1
        if (number_of_iterations > MAX_DEPTH_OF_RECURSION):
            raise Exception(
                "Did not managed to draw correct index")  # this had never happened. But it could for huge population_size

        # redraw the wrong ones
        indices_1[indices_drawn_wrong] = rng.integers(matrix_of_restrictions[0,0], matrix_of_restrictions[1,0],
                                                      size=sum(indices_drawn_wrong))
        indices_2[indices_drawn_wrong] = rng.integers(matrix_of_restrictions[0,1], matrix_of_restrictions[1,1],
                                                      size=sum(indices_drawn_wrong))

    return indices_1, indices_2

def edited_quantile(array, value):
    if len(array) > 0:
        return np.quantile(array, value)
    return 0