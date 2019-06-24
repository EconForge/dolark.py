# Solving Krusell and Smith (1998) with Dolo

In this short notice, we describe the methods used to solve Krusell and Smith (1998) *Income and Wealth Heterogeneity in the Macroeconomy* (JPE)  with and without aggregate productivity shocks. Overall, the resolution relies on an extensive discretization of the state-space. We assume that shocks, agents' policy functions and the law of motion of aggregate capital can be summed up in Markov processes and we solve for the associated transition matrices with fixed-point interative procedures.

## Without aggregate shocks

__krusell_smith_wout_aggregate_shocks.ipynb__ and __aiyagari.yaml__ respectively contain the code for the resolution with dolo and the YAML file associated with the individual program. Households take as given idiosyncratic employment shocks as well as interest rates and wages, while choosing consumption and investment. Interest rates and wages are equal to the marginal productivity of capital and labor. More specifically, households solve

$$
\max_{(c_t)_{t=0}^{+\infty},(a_{t})_{t=1}^{+\infty}} \mathbb E \sum_{t=0}^{\infty} \beta^t u(c_t)
$$

subject to

$$
a_{t+1} + c_t \leq w e_t + (1 + r) a_t,
\quad
c_t \geq 0,
\quad
a_t \geq -B,
\quad
a_0
$$

where

- $ c_t $ is current consumption  
- $ a_t $ is assets  
- $ e_t $ is an exogenous unemployment shock 
- $ w $ is the wage rate  
- $ r $ is the net interest rate  
- $ B $ is the borrowing limit

Firms produce output with capital and labor using a Cobb-Douglas technology.

$$
Y_t = A K_t^{\alpha} N^{1 - \alpha}
$$

where

- $ A $ is a scale parameter
- $ \alpha $ is the capital share in production
- $ K_t $ is aggregate capital  
- $ N $ is total labor supply (which is constant in this simple version of the model)  

Consequently, firms' first order conditions deliver expressions for the net interest rate and the wage

$$
w = A  (1 - \alpha) \left( \frac{K}{N} \right)^{\alpha}\\
r = A \alpha \left( \frac{K}{N} \right)^{\alpha-1}
$$

The resolution procedure consists in finding the aggregate level of capital consistent with both households' offer of savings and firms' capital demand. Accordingly, the corresponding notebook can be decomposed in 2 steps.

1. Pick initial guess for $r$, $w$ and $K$
2. While $K$ evolves more than a given tolerance level
    - Compute the new levels of $r$ and $w$ and update the associated discretized process for exogenous variables.
    - Solve for the individual policy function through the function *time_iteration* and compute the associated Markov chain using *MarkovPolicy*
    - Compute the ergodic distribution of assets and the subsequent level of aggregate capital

# With aggregate shocks

__krusell_smith.ipynb__ and __krusell_smith.yaml__ respectively contain the code for the resolution with dolo and the YAML file associated with the individual program. The model is similar to the  previous case, with the notable introduction of aggregate productivity shocks $z$. This introduces uncertainty over wages and interest rates, which become time-dependent and follow

$$
w_t = A z_t (1 - \alpha) \left( \frac{K_t}{N} \right)^{\alpha}\\
r_t = A z_t \alpha \left( \frac{K_t}{N} \right)^{\alpha-1}
$$

The approximation of the law of motion of aggregate capital becomes a core element of the model resolution and is defined as

$$
K' = \Gamma \left( K, z, z' \right)
$$

In our approach, we introduce a grid for aggregate capital and approximate $\Gamma$ through a Markov transition matrix over this grid. The notebook enforces the following steps

1. Pick a grid for $K$
2. While $\Gamma$ evolves more than a given tolerance level
    - Solve for the individual policy function through the function *improved_time_iteration*
    - Compute the associated Markov chain using *MarkovPolicy*
    - Update the transition matrix for aggregate capital using *UpdateGamma*
    - Update the transitions for exogenous processes using *UpdateTransitions*