# Data

## `lorenz_observations.dat`

Primary dataset for the inference use case: a short synthetic Lorenz-63 trajectory with additive Gaussian noise and known ground-truth parameters

$$
(\sigma, \rho, \beta) = (10, 28, 8/3).
$$

Columns:

```text
t  x  y  z
```

## `legacy_observations.dat`

Archival observational series retained from the original academic assignment. It remains available for exploratory analysis; the documented CLI demo uses the synthetic file above so posterior recovery can be checked against a known truth.
