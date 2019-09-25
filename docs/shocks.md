# Shocks

## Exogenous shocks specification

The type of exogenous shock associated to a model determines the kind of
decision rule, whih will be obtained by the solvers. Shocks can pertain
to one of the following categories: continuous i.i.d. shocks (Normal
law), continous autocorrelated process (VAR1 process) or a discrete
markov chain. The type of the shock is specified using yaml type
annotations (starting with exclamation mark) The exogenous shock section
can refer to parameters specified in the calibration section. Here are
some Examples for each type of shock:

### Normal

For Dynare and continuous-states models, one has to specifiy a
multivariate distribution of the i.i.d. process for the vector of
`shocks` (otherwise shocks are assumed to be constantly 0). This is done
in the distribution section. A gaussian distrubution (only one supported
so far), is specified by supplying the covariance matrix as a list of
list as in the following example.

```yaml
exogenous: !Normal:
    Sigma: [ [sigma_1, 0.0],
            [0.0, sigma_2] ]
```

!!! note
    The shocks syntax is currently rather unforgiving. Normal shocks expect
    a covariance matrix (i.e. a list of list) and the keyword is
    `Sigma`, not `sigma`.


### Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.


```yaml
exogenous: !MarkovChain
    values: [[-0.01, 0.1],[0.01, 0.1]]
    transitions: [[0.9, 0.1], [0.1, 0.9]]
```

It is also possible to combine markov chains together.

```yaml
exogenous: !MarkovTensor:
    - !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.1]]
        transitions: [[0.9, 0.1], [0.1, 0.9]]
    - !MarkovChain
        values: [[-0.01, 0.1],[0.01, 0.1]]
        transitions: [[0.9, 0.1], [0.1, 0.9]]
```
