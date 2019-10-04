# DolARK


## Introduction

DolARK is an experimental project to solve heterogenous agents models with infinitely lived agents. It relies on [Dolo](https://EconForge.github.io/dolo/) to model individual agents behavious and extends its modeling language to describe distributions of agents and aggregate dynamics.

We aim to support the following basic cases:

- heterogenous preferences, i.i.d. idiosyncratic shocks [no aggregate risk](equilibrium.md)
- homogenous preferences, [perturbation](perturbation.md) w.r.t. aggregate risk
- homogenous preferences, dimension reduction of the state-space a la [Krussell-Smith](krussell_smith.md)

## Frequently Asked Questions

No question was ever asked. Certainly because it's all very clear.

## Developper Corner

### Contribute to the documentation

Documentation is contained in the docs subdirectory.
In order to develop it locally:

```
pip install mkdocs
pip install pymdown-extensions
```

Then `mkdocs serve` from within DolARK repository.
On a regular basis, latest version is deployed to github pages [pages](http://www.econforge.org/dolARK/) (for now) using `mkdocs gh-deploy`.

Notebooks are written as Python files and can be opened with Jupyter using the
[jupytext](https://github.com/mwouts/jupytext) extension.
