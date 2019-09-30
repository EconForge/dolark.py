# Shocks


The type of exogenous shock associated to a model determines the kind of
decision rule, which will be obtained by the solvers. Shocks can pertain
to one of the following categories:

- continuous i.i.d. shocks

- continous autocorrelated process (VAR1 process)

- discrete
markov chain.


## Exogenous shocks specification

Exogenous shock processes are specified in the section exogenous . Dolo accepts various exogenous processes such as normally distributed iid shocks, VAR1 processes, and Markov Chain processes.

Here are some examples for each type of shock:

### IID Univariate

....

#### IID Normal

The type of the shock is specified using yaml type annotations (starting with exclamation mark)

Normal distribution with mean mu and variance σ^2 has the probability density function

$$f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$$

A normal shock in the yaml file with mean 0.2 and standard deviation 0.1 can be declared as follows

```
exogenous: !Normal:
    σ: 0.1
    μ: 0.2
```

or

```
exogenous: !Normal:
    sigma: 0.1
    mu: 0.2
```

!!! note
    Greek letter 'σ' or 'sigma' (similarly 'μ' or 'mu' ) are accepted, thanks to the greek translation function.



The exogenous shock section can refer to parameters specified in the calibration section:

```   
symbols:
      states: [a, b]
      controls: [c, d]
      exogenous: [e]
      parameters: [alpha, beta, mu, sigma]

.
.
.

exogenous: !Normal:
      σ: sigma
      μ: mu

```      

#### IID LogNormal

Parametrization of a lognormal random variable Y is in terms of he mean, μ, and standard deviation, σ, of the unique normally distributed random variable X is as follows

$$f(x; \mu, \sigma) = \frac{1}{x \sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(\log(x) - \mu)^2}{2 \sigma^2} \right),
\quad x > 0$$

such that exp(X) = Y

```
exogenous: !LogNormal:
      σ: sigma
      μ: mu

```    

#### Uniform

Uniform distribution over an interval [a,b]

$$f(x; a, b) = \frac{1}{b - a}, \quad a \le x \le b$$


```
symbols:
      states: [k]
      controls: [c, d]
      exogenous: [e]
      parameters: [alpha, beta, mu, sigma, e_min, e_max]

.
.
.

exogenous: !Uniform:
      a: e_min
      b: e_max

```    

#### Beta

If X∼Gamma(α) and Y∼Gamma(β) are distributed independently, then X/(X+Y)∼Beta(α,β).

Beta distribution with shape parameters α and β has the following PDF

$$f(x; \alpha, \beta) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1} (1 x)^{\beta - 1}, \quad x \in [0, 1]$$

```
exogenous: !Beta:
      α: 0.3
      β: 0.1

```    

### Autoregressive process

#### VAR(1)

#### AR(1)


```
exogenous: !AR1
    rho: 0.9
    Sigma: [[σ^2]]
```


### Markov chains

Markov chains are constructed by providing a list of nodes and a
transition matrix.


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

### Product

We can also specify more than one process. For instance if we want to combine a VAR1 and an Normal Process we use the tag Product and write:

```
exogenous: !Product
    p1: !VAR1
         rho: 0.75
         Sigma: [[0.015^2]]

         N: 3

    p2: !Normal:
          σ: sigma
          μ: mu
```

### Mixtures

...

## Discretization methods for continous shocks

To solve a non-linear model with a given exogenous process, one can apply different types of procedures to discretize the continous process:

| Type | Distribution | Discretization procedure             |
|--------------|--------------|-----------------------------------|
|Univariate iid| UNormal(μ, σ)| Equiprobable, Gauss-Hermite Nodes |
|Univariate iid| LogNormal(μ, σ) |Equiprobable |
|Univariate iid| Uniform(a, b ) |Equiprobable|
|Univariate iid| Beta(α, β)   |Equiprobable |
|Univariate iid| Beta(α, β)   |Equiprobable |
| VAR1 |   |Generalized Discretization Method (GDP), Markov Chain |

!!! note
    Here we can define shortly each method. Then perhaps link to a jupyter notebook as discussed: Conditional on the discretization approach, present the results of the corresponding method solutions and simulations. Discuss further discretization methods and related dolo objects.
