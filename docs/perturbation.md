## Perturbation w.r.t. aggregate state.

Implementation restrictions:

- ex ante identical agents
- agents idiosyncratic shocks must be discretizable by markov chains
- aggregate shock is an AR1 (possibly multivariate)

Notations (vague):

- $m_t$ is the aggregate exogenous shock, associated to transition function such that $m_t = \tau(m_{t-1}, \epsilon_t)$
- $x_t$ is a vector representing the values of the decision rule of all agents at date t. Denote by $\varphi^{x_t}()$ the decision rule it determines. Denote by $T$ the time iteration operator which pins down optimal decision today as a function of decisions tomorrow.
- $y_t$ is the vector of aggregate prices
- $\mu_t$ is a vector representing the distributions of agents across endogenous and exogenous idiosyncratic states given $m_t$
- $\Pi(m_t, x_t, y_t)$ is the transition matrix, associated to policy $x_t$ across the individual's states

Recall the definition of the equilibrium function:

$$\int \mathcal{E}\left(e^i, s, x_t(e^i,s), y_t\right) d \mu_t(e^i,s)= 0$$


Knowing the approximate decision rule $\phi^{x_t}$ and distribution $\mu_t$ the above equation is naturally approximated by a function $\mathcal{A}$ such that:


$$\mathcal{A}(m_t, \mu_t, x_t, y_t) = \int \mathcal{E}\left(e^i, s, \varphi^{x_t}(e^i,s), y_t\right) d \mu_t(e^i,s)$$

The whole economy is characterized by the following equations:

- transition ($G$):

$$m_t = \tau(m_{t-1}, \eta_t)$$

$$\mu_t = \mu_{t-1}.\Pi(m_{t-1}, x_{t-1}, y_{t-1})$$

- equilibrium ($F$)

$$x_t = T(m_t, y_t, m_{t+1}, x_{t+1}, y_{t+1})$$

$$\mathcal{A}(m_t, \mu_t, x_t, y_t)=0$$

Note that aggregate states (in the sense that they are predetermined) are $m_t$ and $\mu_t$ that is the aggregate shock and the distribution of agents' states. The controls are $x_t$ and $y_t$ that is agent's decisions and aggregate prices.

For the sake of clarity, let us exemplify the notations above. Considering the Aiyagari (1994) model with aggregate productivity shocks presented in this [note](../single_agent/#ex-ante-identical-agents) from the Model-Definition section
- $s_t = a_t$
- $x_t = i_{t}$
- $y_t = \left( r_t, w_t \right)$
- $e_t^i = \left(y_t,\epsilon_t^i\right)$
- $m_t = z_t$
- $\mu_t$ is the joint distribution of agents over $s_t$ and $\epsilon_t^i$ given $m_t$
    
The transition equation associated with $\tau$ is simply
$$log(z_t) = \rho log(z_{t-1}) + \sigma \eta_t, \quad \eta_t \sim \mathcal{N}(0,1)$$

The equilibrium equations defining $\mathcal{A}$ are
$$
r_t = A \alpha z_t \left( \frac{L}{K_t} \right)^{1 - \alpha} - \delta\\
w_t = A (1-\alpha) z_t \left( \frac{L}{K_t} \right)^{\alpha}\\
K_t = \int \phi^{x_t} \left( \epsilon_t^i, a_t \right) d\mu_t\left( a_t, \epsilon_t^i\right)
$$

$\mathcal T$ is the time-iteration operator, which solves for $x_t$ in the Euler equation given $m_t$, $y_t$, $m_{t+1}$ and $y_{t+1}$