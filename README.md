# Fusion surrogates

A library of surrogate models for tokamak fusion.

This library provides both inference code and model weights and metadata. It is
designed to provide surrogate models for
[TORAX](https://github.com/google-deepmind/torax), but
the models should be usable by other fusion simulators.

## QLKNN_7_11
Currently, this library only holds the QLKNN_7_11 model. This model is a
surrogate of [Qualikiz](https://gitlab.com/qualikiz-group/QuaLiKiz), a
quasilinear gyrokinetic code for turbulent transport in tokamaks.

It is based on the original QLKNN10D model by
[Van de Plassche et al. PoP 2020](https://doi.org/10.1063/1.5134126)
([also available on arxiv](https://arxiv.org/abs/1911.05617)).
It was trained by combining data for the
[QLKNN11D dataset](https://zenodo.org/record/8017522) and
[QLKNN7D-edge dataset](https://zenodo.org/record/8106431). A paper describing
the details of the model is in the works and should be published soon.

The model takes as input:
<ul>
 <li>Normalized logarithmic heat and density gradients for electrons and main
ions ($$R/L_{Te}$$, $$R/L_{Ti}$$, $$R/L_{ne}$$, $$R/L_{ni}$$)</li>
 <li>Safety factor ($$q$$)</li>
 <li>Magnetic shear ($$\hat{s}$$)</li>
 <li>Local inverse aspect ratio ($$r/R_{maj}$$)</li>
 <li>ion-electron temperature ratio ($$T_i/T_e$$)</li>
 <li>Logarithmic ion-electron normalized collisionality ($$\mathrm{log}(\nu^*)$$)</li>
 <li>Normalized density ($$n_i/n_e$$)</li>
</ul>

It outputs ion and electron heat and particle fluxes for each transport mode
(Ion Temperature Gradient [ITG], Electron Temperature Gradient [ETG], Trapped
Electron Modes [TEM]), as well as the the maximum growth rate on ion gyroradius
scales. Specifically, we output the leading flux for that mode
(ion heat flux for ITG, electron heat flux for TEM and ETG),
and ratios of the relevant secondary fluxes to the leading flux of that mode.

More details on the inputs and outputs mentioned above can be found in the
[Qualikiz documentation](https://gitlab.com/qualikiz-group/QuaLiKiz/-/wikis/QuaLiKiz/Input-and-output-variables).

## Installation instructions

### Virtual environment

It is recommended to use a virtual environment to install `fusion_surrogates`.

To install venv:

```shell
pip install --upgrade pip
pip install virtualenv
```

To create and activate a `venv`:

``` shell
python3 -m venv .venv
source .venv/bin/activate
```

Once you are done with your session, you can exit the `venv`:

```shell
deactivate
```

### Installing the library

To install the library:

```shell
pip install fusion_surrogates
```

If you want to run unit tests, install with the `testing` option:

```shell
pip install -e .[testing]
pytest .venv/lib/python*/site-packages/fusion_surrogates
```

## Disclaimer
Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license
at: https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
