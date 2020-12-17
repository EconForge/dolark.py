## Next steps for Dolo/Dolark

Overall focus is to reach a stable usable version of Dolo (aka known as 0.5) and an early version of Dolark with clean, working examples...

### YAML specification:
- test specification introduced so far
- extensions:
  - conditional shocks for agent
  - direct market clearing as in KSM (KS model)
  - forward looking aggregate prices
  - agregate stock variables
- clarify scopes and dependences
- document specs, somehow

### Dolark
- make examples work (add 2d example)
- feasibility matrix for various types of shocks/solution methods
- equilibrium computation
  - performance & algo improvement
- perturbation:
  - dim reduction a la BayerLuetticke
  - performance improvements
  - simulation tools
- perfect foresight simulation
- KSA-like dimension reduction
- special case 1d problems
- work on user interfaces (interactions with simulations)

### Dependences:
- dolo:
  - options system
  - conditional shocks
  - update decision rule object
  - solution methods
  - code cleanup
  - doc, doc, doc
- dolang:
  - support indexed expressions (cosmetically then for real)
  - partial differentiation
  - fix vectorization w.r.t. p
- interpolation
  - custom extrapolation
