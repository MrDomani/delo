.. delo documentation master file, created by
   sphinx-quickstart on Fri Dec 17 20:00:42 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to delo's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api

delo
========

Differential Evolution (DE) optimization algorithms perform satisfactorily even on complex problems in higher dimensionality. However, it is difficult to *a priori* choose optimal parameters.
In this package, we propose **DElo** (DE with adaptation based on Elo rating system). Elo rating, originally used in chess, is a way to measure dynamic fitness.

Installation
------------

Clone repo and run from root:

    `pip install .`

Example 1
---------

.. code-block:: python
   :linenos:

   from delo import DElo, DescribedFunction
   import numpy as np

   def square(x):
       return np.sum(x ** 2, axis=1)
   described_function = DescribedFunction(square, dimension=2,
                                          domain_lower_limit=-10,
                                          domain_upper_limit=10)
   algorithm = DElo(10)
   algorithm.optimize(described_function)

Example 2
---------

.. code-block:: python
   :linenos:

   from delo import DElo, DescribedFunction
   import numpy as np

   def my_single_argument_function(x):
       return np.sum(x ** 2)

   def my_multi_argument_wrapping(x):
       return np.array([my_single_argument_function(xi) for xi in x])

   described_my_function = delo.DescribedFunction(my_multi_argument_wrapping,
                                                  dimension=5,
                                                  domain_lower_limit=-5,
                                                  domain_upper_limit=5)
   algorithm = delo.DElo(100)
   algorithm.optimize(described_my_function, max_f_evals=10000)

Contribute
----------

- Issue Tracker: github.com/MrDomani/delo/issues
- Source Code: github.com/MrDomani/delo


License
-------

The project is licensed under the MIT license.