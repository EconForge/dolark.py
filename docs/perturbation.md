## Perturbation w.r.t. aggregate state.

Implementation restrictions:

- ex ante identical agents
- agents idiosyncratic shocks must be discretizable by markov chains
- aggregate shock is an AR1 (possibly multivariate)

Notations (vague):

- $m_t$ is the aggregate shock, associated to transition function such that $m_t = \tau(m_{t-1}, \epsilon_t)$
- $x_t$ is a vector representing the values of the decision rule of all agents at date t. Denote by $\varphi^{x_t}()$ the decision rule it determines. Denote by $T$ the time iteration operator which pins down optimal decision today as a function of decisions tomorrow.
- $y_t$ is the vector of aggregate prices
- $\mu_t$ is a vector representing the distributions of agents across endogenous and exogenous states
- $\Pi(m_t, x_t, y_t)$ is the transition matrix, associated to policy $x_t$ across the individual's states

Recall the definition of the equilibrium function:

$$\int \mathcal{E}\left(e, s, x_t(e,s), y_t\right) d \mu_t(e,s)= 0$$


Knowing the approximate decision rule $x_t$ and distribution $\mu_t$ the above equation is naturally approximated by a function $\mathcal{A}$ such that:


$$\mathcal{A}(m_t, \mu_t, x_t, y_t) = \int \mathcal{E}\left(e, s, \varphi^{x_t}(e,s), y_t\right) d \mu(e,s)$$

The whole economy is characterized by the following equations:

- transition ($G$):

$$m_t = \tau(m_{t-1}, \epsilon_t)$$

$$\mu_t = \mu_{t-1}.\Pi(m_{t-1}, x_{t-1}, y_{t-1})$$

- equilibrium ($F$)

$$x_t = T(m_t, y_t, m_{t+1}, x_{t+1}, y_{t+1})$$

$$\mathcal{A}(m_t, \mu_t, x_t, y_t)=0$$

Note that aggregate states (in the sense that they are predetermined) are $m_t$ and $\mu_t$ that is the aggregate shock and the distribution of agent's states. The controls are $x_t$ and $y_t$ that is agent's decisions and aggregate prices.
