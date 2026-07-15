# Phase 4.1 — regressed expressions, all testing cases

Fitted parameter values substituted into the best sampled tree (raw tree + parameter dict in the source JSONs).

## v1 (sigma = 1.0, random init)

| target | truth | seed | rec. | NRMSE | regressed expression |
|---|---|---|---|---|---|
| linear | `0.5*x` | 17 | yes | 0.0229 | `0.4886*x` |
| linear | `0.5*x` | 71 | yes | 0.0073 | `0.496352*x` |
| sin_plus_linear | `sin(x) + 0.5*x` | 17 | NO | 0.1453 | `x + sin(x) + cos(x) - 0.7526` |
| sin_plus_linear | `sin(x) + 0.5*x` | 71 | NO | 0.1275 | `-7.2053e-18*sin(x) - cos(x) + (2*sin(x) - cos(x))/x` |
| x_sin | `x*sin(x)` | 17 | yes | 0.0000 | `x*sin(x)` |
| x_sin | `x*sin(x)` | 71 | yes | 0.0000 | `x*sin(x)` |
| rational | `x/(1 + x)` | 17 | NO | 0.2132 | `x/(x + sin(x))` |
| rational | `x/(1 + x)` | 71 | NO | 0.3532 | `x + cos(x) - 1` |
| poly2 | `x**2 - x` | 17 | yes | 0.0065 | `x**2 - 1.0114*x` |
| poly2 | `x**2 - x` | 71 | yes | 0.0030 | `0.994036*x*(x - 0.9938)` |
| freq_sin_atom | `2.3*sin(1.3*x)` | 17 | yes | 0.0094 | `2.3011*sin(1.2973*x)` |
| freq_sin_atom | `2.3*sin(1.3*x)` | 71 | NO | 0.1698 | `2.4264*sin(1.2542*x)` |

## v1 --long (sigma = 1.0, 3x budget)

| target | truth | seed | rec. | NRMSE | regressed expression |
|---|---|---|---|---|---|
| sin_plus_linear | `sin(x) + 0.5*x` | 17 | yes | 0.0153 | `0.4886*x + sin(x)` |
| sin_plus_linear | `sin(x) + 0.5*x` | 71 | NO | 0.1328 | `x/(1.89 - sin(x))` |
| rational | `x/(1 + x)` | 17 | NO | 0.1896 | `0.607300` |
| rational | `x/(1 + x)` | 71 | NO | 0.1854 | `0.627800` |
| freq_sin_atom | `2.3*sin(1.3*x)` | 17 | yes | 0.0094 | `2.3011*sin(1.2973*x)` |
| freq_sin_atom | `2.3*sin(1.3*x)` | 71 | yes | 0.0086 | `2.2893*sin(1.2981*x)` |

## v2 (corrected protocol)

| target | truth | seed | rec. | NRMSE | regressed expression |
|---|---|---|---|---|---|
| linear | `0.5*x` | 17 | yes | 0.0229 | `0.4886*x` |
| linear | `0.5*x` | 71 | yes | 0.0073 | `0.4963*x` |
| sin_plus_linear | `sin(x) + 0.5*x` | 17 | yes | 0.0171 | `0.487232*x + sin(x)` |
| sin_plus_linear | `sin(x) + 0.5*x` | 71 | yes | 0.0015 | `0.4989*x + sin(x)` |
| x_sin | `x*sin(x)` | 17 | yes | 0.0000 | `x*sin(x)` |
| x_sin | `x*sin(x)` | 71 | yes | 0.0000 | `x*sin(x)` |
| rational | `x/(1 + x)` | 17 | yes | 0.0324 | `x/(x + 1.1685)` |
| rational | `x/(1 + x)` | 71 | yes | 0.0034 | `x/(x + 1.017)` |
| poly2 | `x**2 - x` | 17 | yes | 0.0000 | `x**2 - x` |
| poly2 | `x**2 - x` | 71 | NO | 0.5654 | `x**2` |
| freq_sin_atom | `2.3*sin(1.3*x)` | 17 | yes | 0.0094 | `2.3011*sin(1.2973*x)` |
| freq_sin_atom | `2.3*sin(1.3*x)` | 71 | yes | 0.0086 | `2.2893*sin(1.2981*x)` |

## depth-3 suite (corrected protocol, 400 rounds)

| target | truth | seed | rec. | NRMSE | regressed expression |
|---|---|---|---|---|---|
| poly3 | `x**3 - 2*x` | 17 | yes | 0.0018 | `x**2*(x - 2.0114/x)` |
| poly3 | `x**3 - 2*x` | 71 | yes | 0.0008 | `0.9994*x**3 - 2*x` |
| inv_quad | `3*x/(1 + x**2)` | 17 | NO | 0.0692 | `-(0.3608 - sin(x))*(sin(x) + cos(x))/(x + 0.8285) + 0.4249*sin(x) + 0.7348` |
| inv_quad | `3*x/(1 + x**2)` | 71 | yes | 0.0046 | `0.5906/(0.198358*x + 0.1957/x)` |
| mixed | `x*sin(x) + x/(1 + x)` | 17 | yes | 0.0166 | `0.9275*x*sin(x) + 0.7139 - 0.423567*sin(x)*cos(x)/x` |
| mixed | `x*sin(x) + x/(1 + x)` | 71 | yes | 0.0071 | `-0.0862*x*(-sin(x) - 3.5174) + x*sin(x) + 0.2169*cos(x)` |
