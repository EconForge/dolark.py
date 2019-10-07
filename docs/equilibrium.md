<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@3"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@5"></script>

## Time-invariant equilibrium

Recall the model written in section X. The agents decision depends on the values processes, which
are pinned down by aggregate conditions. These conditions are fixed in an additional equilibrium
section.

Let us start from an example:

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

exogenous: !AR1
    ρ: ρ
    σ: σ

equilibrium:
    K = k
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
    β: !Uniform
        a: 0.95
        b: 0.96
```
