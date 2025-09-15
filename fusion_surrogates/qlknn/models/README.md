# QLKNN_7_11

## Datasets

QLKNN_7_11 was trained on a combination of two datasets:

 - QLKNN11D (https://zenodo.org/record/8017522)
 - QLKNN7D-edge (https://zenodo.org/record/8106431)

## Filtering

We used similar filtering techniques as described in [Fast modeling of turbulent transport in fusion plasmas using neural networks](https://pubs.aip.org/aip/pop/article-abstract/27/2/022310/1062590/Fast-modeling-of-turbulent-transport-in-fusion?redirectedFrom=fulltext).

Datapoints were rejected if:

 - convergence was not achieved
 - total heat flux is negative
 - difference between total particle flux and derived particle flux from diffusion and convection transport coefficients is more than 50%
 - difference between unweighted sum of ITG + TEM mode contributions, and self-consistent total flux calculation, was more than 50%
 - ambipolarity was violated by more than 50%
 - any transport coefficient is non-zero but predicted to be smaller than 10−4 in GyroBohm (GB) units
 - data point fluxes are outliers, defined as the lower a upper 2-percentile of the unfiltered dataset.

## Input ranges

The datasets sampled an hyper-grid of the input space. See [Qualikiz input variables definitions](https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input%20and%20output%20variables#input). Here are the minima and
maxima of the inputs in each datasets:

- QLKNN11D:

  ```
  {'Ati': {'min': 9.9999998245167e-15, 'max': 16.0},
  'Ate': {'min': 9.9999998245167e-15, 'max': 16.0},
  'Ane': {'min': -5.0, 'max': 5.0},
  'Ani': {'min': -15.0, 'max': 15.0},
  'q': {'min': 0.6600000262260437, 'max': 10.0},
  'smag': {'min': -1.0, 'max': 4.0},
  'x': {'min': 0.10000000149011612, 'max': 0.949999988079071},
  'Ti_Te': {'min': 0.24999983608722687, 'max': 2.5000014305114746},
  'LogNuStar': {'min': -5.000293731689453, 'max': -0.0002939051773864776},
  'normni': {'min': 0.699999988079071, 'max': 1.0}
  }
  ```

- QLKNN7D:

  ```
  {'Ati': {'min': 5.0, 'max': 150.0},
  'Ate': {'min': 5.0, 'max': 150.0},
  'Ane': {'min': 2.0, 'max': 110.0},
  'Ani': {'min': 2.0, 'max': 110.0},
  'q': {'min': 2.0, 'max': 30.0},
  'smag': {'min': 1.0, 'max': 40.0},
  'x': {'min': 0.949999988079071, 'max': 0.949999988079071},
  'Ti_Te': {'min': 1.0, 'max': 1.0},
  'LogNuStar': {'min': -1.0002938508987427, 'max': 0.4768274426460266},
  'normni': {'min': 0.5, 'max': 1.0}
  }
  ```

## Output ranges:
  See [Qualikiz output variables definitions](https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input%20and%20output%20variables#output). Here are the ranges for the outputs (both datasets combined):

  ```
  {
    'efiITG': {'min': 2.86e-01, 'max': 442.27},
    'efeITG': {'min': 1.92e-03, 'max': 446.69},
    'pfeITG': {'min': -1.33e02, 'max': 132.68},
    'efeTEM': {'min': 1.68e-01, 'max': 134.78},
    'efiTEM': {'min': 9.88e-04, 'max': 257.43},
    'pfeTEM': {'min': -8.90e01, 'max': 88.95},
    'efeETG': {'min': 1.41e-01, 'max': 73.15},
    'gamma_max': {'min': 0.0, 'max': 0.4599998891353607}
  }
  ```

