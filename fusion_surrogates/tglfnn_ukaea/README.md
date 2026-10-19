# TGLFNN-UKAEA

JAX implementation of the UKAEA TGLF surrogate. Different versions of the
surrogate exist, including a *multi-machine* version and a *spherical tokamak*
version.

Weights, normalisations, config files and compiled models are stored at
[ukaea/tglfnn-ukaea](https://github.com/ukaea/tglfnn-ukaea). The parameters
used here are exported from the ONNX models in that repository, which this
implementation reproduces. The network version can be selected with
`TGLFNNukaeaModel(machine, version=...)`, and otherwise defaults to:

| `machine` | Default version | Model |
|---|---|---|
| `multimachine` | 3.2.1 | MultiMachineHyper_1Aug25 |
| `step` | 1.0.1 | JETTO-TGLF-EM-17Feb25 |