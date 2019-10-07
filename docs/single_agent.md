# Model specification

## Single-Agent Program



Recall the setup of a single-agent problem, as solved by
[dolo](https://github.com/EconForge/dolo): agent $i$ attempts to control a stochastic process, driven by exogenous process $m_t$, by choosing decisions $x_t$ so that the evolution of the endogenous state is:

$$s_t = g(e_{t-1}, s_{t-1}, x_{t-1}, e_t)$$

where $g$ is a known function. Since $m_t$ follows a markov chain, $m_{t+1}$ is known using m_t and the arbitrage equation is written:

$$E_{e_{t+1}|e_t} = \left[ f(e_t, s_t, x_t, e_{t+1}, s_{t+1}, x_{t+1})\right]$$

for some known function $f$, called arbitrage function. Optimal choice is function $\varphi()$, such that $x_t=\varphi(e_t, s_t)$.

!!! note

    For a basic consumption savings problem, endogenous state is level of assets $a_t$, the control the re-invested amount $i_t$ so that that the law of motion of asset holdings is:

    $$a_t = (i_{t-1})(1+r_t) + exp(\epsilon_t)w_t$$

    where $\epsilon_t$ is an idiosyncratic process for efficiency and $r_t$ (resp. $w_t$) a process for interest rate (resp. wage.).

    The optimality condition is:

    $$β E_t \left[ \frac{U^{\prime}(\overbrace{a_{t+1}-i_{t+1}}^{c_{t+1}})}{U^{\prime}(\underbrace{a_{t}-i_{t}}_{c_t})}r_{t+1}\right]=1$$

    For a traditional one agent problem, $r_t$ and $w_t$ are typically equal to $\overline{r}$ and $\overline{w}$ and $\epsilon_t$ an AR1.

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


In Dolark, the behaviour of each agent is defined by the exact same functions $f$ and $g$, with a small modification to the way the exogenous process is modeled. We assume there is a state of the world $ω_t$, whose stochastic process is known to the agent, and a projection function $p()$ which defines relevant exogenous states of the agent: $m_t=p(ω_t)$. Since $ω_t$ is required to predict $\omega_{t+1}$, the optimality condition becomes:

$$E_{\color{\red}{ω_{t+1}}|\color{\red}{ω_t}} = \left[ f(m_t, s_t, x_t, m_{t+1}, s_{t+1}, x_{t+1})\right]$$

or

$$E_{\color{\red}{ω_{t+1}}|\color{\red}{ω_t}} = \left[ f(p(\color{\red}{ω_t}), s_t, x_t, p(\color{\red}{ω_{t+1}}), s_{t+1}, x_{t+1})\right]$$

Optimal choice is now a function $\varphi()$, such that $x_t=\varphi(\color{\red}{ω_t}, s_t)$. The construction of $\omega_t$ and the exact specification of $p()$ is a result from the market equilibrium solution procedure and is typically never constructed explicitly.


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
    Note that it is not mandatory to use a `!ConstantProcess`. Any type of process could be used
    here. It is merely a place holder and the aggregate solution procedure, will seek to replace it by a suitable process, for instance a Markov Chain or a perturbed process.

The rest of the YAML file writes following the [documentation of dolo](https://dolo.readthedocs.io/en/latest/). See the full source below.

??? note "Ayiagari (1994): consumption savings model"

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

### Equilibrium conditions

Consider now a continuum of mass 1, made of agents that are all identical ex-ante.  Consider and aggregate process $m_t$ and some aggregate equilibrium variables $y_t$. Denote by $\mu_t$ the distribution of the endogenous states across all agents and by $x_t$ the choices made by said agents (the mathematical nature of these objects is left a bit fuzzy for now). The whole economy is specified by:

1. a law of motion for process $m_t$
2. a projection function p(.) whose output matches the aggregate exogenous process taken into account by individual agents
3. equilibrium conditions defined by a function $\mathcal{A}$ such that

    $$\mathcal{A}\left(m_t, μ_t, x_t, y_t\right) = 0$$

??? alert

    Not happy at all with these notations.

!!!note "An Example: Aiyagari (1994)"

    Here is how wage and interest rate is determined for all agents.
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

    It is not necessary to redefine variables that were defined in the agent's program. All these variables are recognized and implicitly indexed. For instance `a`, which was defined in the preceding section, implicitly stands for $a_i$.

    Also, the equilibrium condition implicitly features an integration over all agents, so that `K=k` is intepreted as:

    $$K_t = \int_i a_{i,t}$$
