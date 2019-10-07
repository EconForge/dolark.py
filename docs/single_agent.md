## Single-Agent Program

Given a vector of aggregate endogenous variables (e.g. wages or interest rates), the program of the single-agent can be solved using traditional methods in computational economics. In DolARK, we use [dolo](https://github.com/EconForge/dolo) to do so. At this point, the only relevant question concerns the proper writing of the YAML file including these aggregate endogenous variables.

### The YAML file

To be more specific, the introduction of aggregate endogenous variables amends the YAML file in two steps:

1. Aggregate endogenous variables are first declared in the `exogenous` subsection of the `symbols` section.
2. We then specify an additional `!ConstantProcess` entry in the `exogenous` section to initalize the behavior of aggregate endogenous variables.

The rest of the YAML file writes following the [documentation of dolo](https://dolo.readthedocs.io/en/latest/)

### An Example: Aiyagari (1994)

Consider the model proposed by Aiyagari (1994). The economy includes one homogeneous good, which is produced with labor and capital. There exists a continuum of households who inelastically supply labor, receive wage $w$ and face idiosyncratic employment shocks $e$. Households may consume $c$ and save $a$, which yields interests at rate $r$. Households are credit-constrained ; they cannot borrow beyond a cap $\underline{a}$. An household's value function verifies

$v_t(k_t) = \max_{c_t} u(c_t) + \beta \mathbb E_t v_{t+1}(k_{t+1})$

subject to
$$c_t + a_{t+1} =  (1+r) a_t + e_t w \quad \text{and} \quad a \geq \underline{a}$$

Firms in perfect competition produce the homogeneous good with a Cobb-Douglas technology and choose inputs to maximize profits. Thus, the representative firm's program is

$$\max_{K_t,L} A K^\alpha L^{1-\alpha} - (r+\delta) K_t - w L$$

where
- $K_t$ is aggregate capital
- $L$ is aggregate labor supply
- $\delta$ is the depreciation rate of capital
- $A$ is the scale factor of production
- $\alpha$ is the output elasticity w.r.t capital

The first-order conditions of the firm's program deliver expressions for $r$ and $w$.

$$
r = A \alpha  \left( \frac{L}{K} \right)^{1 - \alpha} - \delta\\
w = A (1-\alpha) \left( \frac{L}{K} \right)^{\alpha}
$$

In this example, the aggregate endogenous variables are the interest rate $r$ and the wage $w$. The two modifications stated in the previous paragraph appear
1. in the `exogenous` subsection of the `symbols` section looks like
```   
    symbols:
    ...
        exogenous: [r,w,e]
    ...
```
2. in the `exogenous` section
```
    exogenous:
    ...
        r,w: !ConstantProcess
            μ: [r, w]
    ...
```

Overall, , assuming that logged $e$ follow an AR1 with persistence $\rho = 0.95$ and standard deviation $\sigma = 0.06$, the corresponding YAML file looks like

```
    symbols:
        states: [a]
        exogenous: [r, w, e]
        parameters: [alpha, L, delta, beta, a_min, a_max]
        controls: [i]


    definitions:
        c: (1+r)*a+w*exp(e) - i

    equations:

        arbitrage:
            - 1-beta*(1+r(+1))*c/c(+1) | -B <= i <= (1+r)*a+w*exp(e)

        transition:
            - a = i(-1)

    calibration:
        ...
        r: alpha*(L/K)**(1-alpha) - delta
        w: (1-alpha)*(K/L)**(alpha)
        ...
    domain:
        a: [a_min, a_max]

    exogenous:
        r,w: !ConstantProcess
            μ: [r, w]
        e: !VAR1
            ρ: 0.95
            Σ: [[0.06**2]]

    options:
        grid:
            !Cartesian
            orders: [30]
```
