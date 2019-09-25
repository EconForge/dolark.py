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

### IID Univariate

| Distribution | Probability density function | Discretization Method             |
|--------------|------------|-----------------------------------|
| UNormal(μ, σ)| $$ f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right) $$ |Equiprobable, Gauss-Hermite Nodes  |Univariate iid|
| LogNormal(μ, σ)    | $$     f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}}\exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right),\quad x > 0 $$       |Equiprobable |
| Uniform(a, b )      |$$ f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b $$|Equiprobable|
| Beta(α, β)         |$$ f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 x)^{\beta - 1}, \quad x \in [0, 1]$$ |Equiprobable |


#### IID Normal

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

Normal(μ, σ) creates a Normal distribution with mean mu and variance σ^2 with probability density function

$$f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$$



!!! note
    The shocks syntax is currently rather unforgiving. Normal shocks expect
    a covariance matrix (i.e. a list of list) and the keyword is
    `Sigma`, not `sigma`.

#### LogNormal

$$f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right),
\quad x > 0$$

#### Uniform

$$f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b$$

#### Beta

$$f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 x)^{\beta - 1}, \quad x \in [0, 1]$$


Beta(a, b) # Beta distribution with shape parameters a and b

### Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.


$$μ= \rightarrow$$

```
```








```
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
