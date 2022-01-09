# Differenital Evolution with Elo-based adaptation
[![Documentation Status](https://readthedocs.org/projects/delo/badge/?version=latest)](https://delo.readthedocs.io/en/latest/?badge=latest) <br>
[documentation](https://delo.readthedocs.io/en/latest/index.html)
## Overview
Differential Evolution (DE) optimization algorithms perform satisfactorily even on complex problems in higher dimensionality. However, it is difficult to *a priori* choose optimal parameters.
In this package, we propose **DElo** (DE with adaptation based on Elo rating system). Elo rating, originally used in chess, is a way to measure dynamic fitness.

## Installation
Navigate to repository root folder and execute command
```
pip install -e .
```
That installs a **developer** version - any changes to files in package source will immediately take effect. No reintalls required.

To install a **regular** version, just execute without `-e` option.

## Example
```
import delo
```

If one have a function that takes a single argument and returns a single value, like this:
```
def my_single_argument_function(x):
    return np.sum(x ** 2)
```
one have to wrap it like this:
```
def my_multi_argument_wrapping(x):
    return np.array([my_single_argument_function(xi) for xi in x])

described_my_function = delo.DescribedFunction(my_multi_argument_wrapping,
                                               dimension=5,
                                               domain_lower_limit=-5,
                                               domain_upper_limit=5)
```
Then, one can use DElo to find the optimal value of the function:
```
algorithm = delo.DElo(100)
solution, best_f_value = algorithm.optimize(described_my_function, max_f_evals=10000)
print(solution, best_f_value)
# [0.0  0.0 -0.0  0.0  0.0], 8.5e-09
```
Or one can use DEloTQI to find the optimal value of the function:
```
algorithm = delo.DElo_ties_and_QI(100)
solution, best_f_value = algorithm.optimize(described_my_function, max_f_evals=10000)
print(solution, best_f_value)
# [0.0 -0.0 -0.0 -0.0  0.0], 1.1e-11
```

## References:
1. [SHADE](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.5259&rep=rep1&type=pdf)
2. "For a more comprehensive introduction to ES, see Beyer and Schwefel (2002)" ~ [preprint from 2021: "Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges"](https://www.researchgate.net/publication/353234152_Hyperparameter_Optimization_Foundations_Algorithms_Best_Practices_and_Open_Challenges?pli=1&loginT=7g6vBIQMadxoexmLGqhqYgf_hbU7syYOMK2fVRg8NuujDPL6zUglx3nMuG4grxh27pcimvyCLP3fk9K7kqieWvrC4agyDrs5FQ&uid=UYtHAAH0ScOSPfHCn0vHrwlgRHalOpRtqDfj&cp=re442_pb_hnsg_naas_p113&ch=reg&utm_medium=email&utm_source=researchgate&utm_campaign=re442&utm_term=re442_pb_hnsg_naas&utm_content=re442_pb_hnsg_naas_p113)
3. Beyer, H.-G., & Schwefel, H.-P. (2002). Evolution strategies - A comprehensive introduction. Natural Computing, 1, 3–52.
[springer link](https://link.springer.com/article/10.1023/A:1015059928466)

## ELO system for chess explained:
1. https://youtu.be/AsYfbmp0To0
2. https://en.wikipedia.org/wiki/Elo_rating_system

The general idea is when comparing two solutions/genomes/specimens. If one has a better score do not consider it better, but rather it will be the one with a bigger probability of being better.

It is based on <img src="https://render.githubusercontent.com/render/math?math=S(f(x_1)-f(x_2))"> where S is sigmoid function <img src="https://render.githubusercontent.com/render/math?math=S(y) = \frac{1}{1 %2B e^{-y}}">, <img src="https://render.githubusercontent.com/render/math?math=f"> is score function, <img src="https://render.githubusercontent.com/render/math?math=x_i"> being i-th specimen.

## Acknowledgements
Developped as part of joint Engineer's Thesis of Przemyslaw Adam Chojecki and Pawel Morgen under supervision of Michal Okulewicz, Ph.D. at Warsaw University of Technology.
