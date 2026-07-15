# Feynman suite — regressed expressions (gauss dropped by design)

| law | Feynman ID | seed | rec. | NRMSE | regressed expression |
|---|---|---|---|---|---|
| flux | I.12.2 | 17 | yes | 0.0064 | `1.9871/x**2` |
| flux | I.12.2 | 71 | yes | 0.0000 | `2/x**2` |
| scattering | III.17.37 | 17 | yes | 0.0126 | `0.8542*cos(x) + 1.1864` |
| scattering | III.17.37 | 71 | yes | 0.0090 | `0.8541*cos(x) + 1.1932` |
| velocity_add | I.16.6 | 17 | yes | 0.0060 | `(x + exp(x) - sin(x) + cos(x) - exp(-x)*cos(x))/(exp(x) + cos(x))` |
| velocity_add | I.16.6 | 71 | yes | 0.0065 | `(x + exp(x))/(exp(x) + 1.0382)` |
| polarization | II.11.28 | 17 | no | 0.1741 | `-2.0468 - 9.0564/(x - 3.0013)` |
| polarization | II.11.28 | 71 | yes | 0.0014 | `2.98329*x/(2.9967 - x) + 1` |
| planck | I.41.16 | 17 | yes | 0.0235 | `0.853898*x**2*(x + 0.7191)*exp(-x)` |
| planck | I.41.16 | 71 | yes | 0.0153 | `0.7419*x**2*(x + 1.9293)/(x + exp(x))` |
