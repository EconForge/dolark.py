<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@3"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@5"></script>

## Time-invariant equilibrium

Recall the model written in section X. The agents decision depends on the values processes, which
are pinned down by aggregate conditions. These conditions are fixed in an additional equilibrium
section.

Let us start from an example:

```yaml tab=
symbols:
    exogenous: [z]
    aggregate: [K]
    parameters = [A, alpha, delta, ρ]
)

calibration:
    A: 1
    alpha: 0.36
    delta: 0.025
    K: 40
    z: 0
    ρ: 0.95

exogenous: !AR1
    ρ: ρ
    σ: σ

equilibrium:
    K = k
```

```python tab=
class KrussellSmith(AggregateModel):

    symbols = dict(
        exogenous = ["z"],
        aggregate = ["K"],
        parameters = ["A", "alpha", "delta", 'ρ']
    )

    calibration_dict = dict(
        A = 1,
        alpha = 0.36,
        delta = 0.025,
        K = 40,
        z = 0,
        ρ = 0.95
    )

    def τ(self, m, p):
        # exogenous process is assumed to be deterministic
        ρ = p[3]
        return m*ρ

    def definitions(self, m: 'n_e', y: "n_y", p: "n_p"):
        from numpy import exp
        z = m[0]
        K = y[0]
        A = [0]
        alpha = p[1]
        delta = p[2]
        N = 1
        r = alpha*exp(z)*(N/K)**(1-alpha) - delta
        w = (1-alpha)*exp(z)*(K/N)**(alpha)
        return {'r': r, "w": w}

    def 𝒜(self, m0: 'n_e', μ0: "n_m.N" , xx0: "n_m.N.n_x", y0: "n_y", p: "n_p"):

        import numpy as np
        kd = sum( [float((μ0[i,:]*xx0[i,:,0]).sum()) for i in range(μ0.shape[0])] )
        aggres_0 = np.array( [kd - y0[0] ])
        return aggres_0
```

Graphical represenation


<div id="view"></div>
<script>vegaEmbed('#view','../graphs/distrib.json');</script>

## heterogeneity


Now, a model with idiosyncratic heterogeneity:



```yaml
symbols:
    exogenous: [z]
    aggregate: [K]
    parameters = [A, alpha, delta, ρ]
)

calibration:
    A: 1
    alpha: 0.36
    delta: 0.025
    K: 40
    z: 0
    ρ: 0.95

exogenous: !AR1
    ρ: ρ
    σ: σ

equilibrium:
    K = k

heterogeneity:
    β: Uniform:
        a: 0.95
        b: 0.96
```
