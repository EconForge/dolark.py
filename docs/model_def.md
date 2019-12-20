# Model Definition

The definition of the model involves two YAML files: the first one concerns the individual agents while the second one includes information about the equilibrium to close the model.

## Single-agent problem

Recall the setup of a single-agent problem, as solved by
[dolo](https://github.com/EconForge/dolo): an agent attempts to control a stochastic process, driven by exogenous process $e_t^i$, by choosing decisions $x_t$ so that the evolution of the endogenous state is:

$$s_t = g(e_{t-1}^i, s_{t-1}, x_{t-1}, e_t^i)$$

where $g$ is a known function. Since $e_t^i$ follows a markov chain, $e_{t+1}^i$ is known using $e_t^i$ and the arbitrage equation is written:

$$E_{e_{t+1}^i|e_t^i} = \left[ f(e_t^i, s_t, x_t, e_{t+1}^i, s_{t+1}, x_{t+1})\right]$$

for some known function $f$, called arbitrage function. Optimal choice is function $\varphi()$, such that $x_t=\varphi(e_t^i, s_t)$.

The exogenous process $e_t^i$ contains all the necessary information for the agent to optimize its decisions, namely
- idiosyncratic shocks denoted $\epsilon_t^i$
- aggregate prices $y_t$.

!!!note

    For a basic consumption savings problem, the endogenous state $s_t$ is the level of assets $a_t$, while the control $x_t$ is the re-invested amount $i_t$ so that that the law of motion of asset holdings is:

    $$a_t = (i_{t-1})(1+r_t) + exp(\epsilon_t)w_t$$

    where $\epsilon_t^i$ is an idiosyncratic process for efficiency and $r_t$ (resp. $w_t$) a process for interest rate (resp. wage.). Fitting the notations above, $y_t = \left(r_t, w_t \right)$ and $e_t^i = \left( \epsilon_t^i, y_t \right)$

    The optimality condition is:

    $$β E_t \left[ \frac{U^{\prime}(\overbrace{a_{t+1}-i_{t+1}}^{c_{t+1}})}{U^{\prime}(\underbrace{a_{t}-i_{t}}_{c_t})}r_{t+1}\right]=1$$

    For a traditional one agent problem without aggregate shocks, $r_t$ and $w_t$ are typically constant and equal $\overline{r}$ and $\overline{w}$ while $\epsilon_t^i$ follows an AR(1).

    1. Since $\overline{w}$ and $\overline{r}$ are constant, they can be declared as parameters in the `symbols` section of the model:
    ```   
    symbols:
        states: [a]
        exogenous: [e]
        parameters: [β, γ, ρ, σ, w, r]
    ```
    2. In the `exogenous` section
    ```
    exogenous:
        e: !AR1
            ρ: 0.9
            σ: 0.01
    ```

In DolARK, the behavior of each agent is defined by the exact same functions $f$ and $g$ as in dolo, with a small modification to the way the exogenous process is modeled.

We assume there is a state of the world $ω_t^i$, whose stochastic process is known to the agent, and a projection function $p()$ which defines relevant exogenous states of the agent: $e_t^i=p(ω_t^i)$. Since $ω_t^i$ is required to predict $\omega_{t+1}^i$, the optimality condition becomes:

$$E_{\color{\red}{ω_{t+1}^i}|\color{\red}{ω_t^i}} = \left[ f(e_t^i, s_t, x_t, e_{t+1}^i, s_{t+1}, x_{t+1})\right]$$

or

$$E_{\color{\red}{ω_{t+1}^i}|\color{\red}{ω_t^i}} = \left[ f(p(\color{\red}{ω_t^i}), s_t, x_t, p(\color{\red}{ω_{t+1}^i}), s_{t+1}, x_{t+1})\right]$$

Optimal choice is now a function $\varphi()$, such that $x_t=\varphi(\color{\red}{ω_t^i}, s_t)$. The construction of $\omega_t^i$ and the exact specification of $p()$ is a result from the market equilibrium solution procedure and is typically never constructed explicitly.


The bottom line is that the agent's problem, and how to solve it, is essentially unchanged, save for the nature of the exogenous process. As a result, the one agent problem can be written using the dolo conventions, and the original routines are run unmodified.

Variables appearing in the agent's program, which are determined by market interaction, must be defined as exogenous shocks. By convention these aggregate variables, must come before idiosyncratic shocks.


!!! note

    In this example, the aggregate endogenous variables are the interest rate $r$ and the wage $w$. The two modifications stated in the previous paragraph appear below

    1. in the `exogenous` subsection of the `symbols` section `r` and `w` are declared as exogenous, and declared *before* the idiosyncratic shocks `e`.
    ```   
    symbols:
        ...
        exogenous: [r,w,e]
    ...
    ```
    2. in the `exogenous` section
    ```
    exogenous:
        r,w: !ConstantProcess
            μ: [r, w]
        e: !AR1
            ρ: 0.9
            σ: 0.01
    ```

!!! warning
    Note that it is not mandatory to use a `!ConstantProcess` for shocks determined at the aggegate level. Any type of process could be used
    here. It is merely a place holder and the aggregate solution procedure, will seek to replace it by a suitable process, for instance a Markov Chain or a perturbed process.

The rest of the YAML file writes following the [documentation of dolo](https://dolo.readthedocs.io/en/latest/). See the full source below.

??? note "Aiyagari (1994): consumption savings model"

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

## Equilibrium conditions

The second YAML file states the equilibrium conditions.

### Ex ante identical agents

Consider now a continuum of mass 1, made of agents that are all identical ex-ante. Consider an aggregate process $m_t$ and some aggregate equilibrium variables $y_t$ (also called abusively "prices"). Given $m_t$, denote by $(\epsilon^i,s)\rightarrow\mu_t(\epsilon^i,s)$ the joint distribution of the endogenous states and idiosyncratic exogenous states across all agents and by $x_t$ the choices agents make (the mathematical nature of these objects is left a bit fuzzy for now). The whole economy is specified by:

1. A law of motion for process $m_t$
2. A projection function p(.) whose output *at least* includes the aggregate exogenous process $y_t$
3. Equilibrium conditions defined by a function $\mathcal{E}$ such that

    $$\int \mathcal{E}\left(e_t, s_t, x(e_t,s_t), y_t\right) d \mu_t (e,s)= 0$$

where integral is taken over the distribution of agents over all their endogenous and idiosyncratic exogenous states.

!!! note

    To link the current notations to the paragraph above, $\omega_t^i$ consists in the vector $\left(\epsilon_t^i, m_t, \mu_t \right)$.
    Generally, if idiosyncratic shocks $\epsilon_t^i$ are independent of the aggregate exogenous process $m_t$, the output of function $p$ needs not to include $\epsilon_t^i$ and the corresponding exogenous process is directly defined in the single-agent YAML file. It may be different if the process of $\epsilon_t^i$ depends on $m_t$, as is the case in Krusell and Smith (1998)'s specification.

!!! warning

    In this formulation, $m_t$ and $d \mu_t$ are aggregate "states", that is they are predetermined and $x_t,y_t$ are aggregate "controls", meaning there is a function $\Phi()$ such that $(x_t,y_t)=\Phi(m_t,\mu_t)$.

    the formulation above implies some restrictions that might be lifted in the future:

    - there is no persistent aggregate endogenous state variable
    - equations pinning down the aggregate equilibrium are static. They cannot define forward looking prices.

!!!note "An Example: Aiyagari (1994) with aggregate productivity shocks"

    Here is how wage and interest rate is determined for all agents.
    Firms in perfect competition produce the homogeneous good with a Cobb-Douglas technology and choose inputs to maximize profits. Thus, the representative firm's program is

    $$\max_{K_t,L} z_t A K_t^\alpha L^{1-\alpha} - (r_t+\delta) K_t - w_t L$$

    where

    - $z_t$ is the aggregate productivity shock
    - $K_t$ is aggregate capital
    - $L$ is aggregate labor supply
    - $\delta$ is the depreciation rate of capital
    - $A$ is the scale factor of production
    - $\alpha$ is the output elasticity w.r.t capital

    The first-order conditions of the firm's program deliver expressions for $r_t$ and $w_t$.

    $$
    r_t = A \alpha z_t \left( \frac{L}{K_t} \right)^{1 - \alpha} - \delta\\
    w_t = A (1-\alpha) z_t \left( \frac{L}{K_t} \right)^{\alpha}
    $$

    While the YAML file for the single-agent is built as described above, the YAML file for the aggregate part writes as follows if $z_t$ follows a logged AR(1).

    ```yaml
    symbols:
        exogenous: [z]
        aggregate: [K]
        parameters: [A, alpha, delta, ρ]

    calibration:
        A: 1
        alpha: 0.36
        delta: 0.025
        K: 40
        z: 0
        ρ: 0.95
        σ: 0.025

    exogenous: !AR1
        ρ: ρ
        σ: σ

    projection:
        r: alpha*exp(z)*(N/K)**(1-alpha) - delta
        w: (1-alpha)*exp(z)*(K/N)**(alpha)

    equilibrium:
        K = a
    ```

    Note the familiarity with the one-agent problem: symbols, calibration of parameters, and definition of exogenous shocks, are defined in exactly the same way.

    It is not necessary to redefine variables that were defined in the agent's program. All these variables are recognized and implicitly indexed. For instance `a`, which was defined in the preceding section and is implicitly with the agents distribution.

    Also, the equilibrium condition implicitly features an integration over all agents, so that `K=a` is intepreted as:

    $$K_t = \int_{e,s} a(e,s) dμ_t(e,s)$$

### Ex ante heterogeneous agents

DolARK allows to define models where one or many parameters of one agent's program
is heterogenous, and distributed after some distribution. This is done by adding a special `distribution` section.

!!! note

    To model heterogeneity in the discount factor, add a distribution section:
    ```
    distribution:
        β: !Uniform
            a: 0.95
            b: 0.96
    ```

    The effect of this section will be to instruct dolark to solve for several agents problem for different values  $\beta^i$ associate with probabilities $\pi^i$ that approximate distribution $\beta$. Denote by $a^{i_\beta}$ and $d\mu^{i_{\beta}}$ the corresponding decision rules and ergodic distributions. The market clearing condition becomes:


    $$K_t = \int_{β} dπ(β) \int_{e,s} a^{β}(e,s) dμ^{β}_t(e,s)$$


??? attention

    There is a second experimental way to model heterogeneity: to treat the heterogenous parameter
    as a persistent shock. To implement it, the agent's problem needs to feature this shock as the first exogenous shock. Note that it can create subtle timing inconsistencies in the model. For instance symbol `\beta` would be taken as representing `β_t` which might not be allowed for certain equations specifications. (transitions and arbitrage equations should be fine)
