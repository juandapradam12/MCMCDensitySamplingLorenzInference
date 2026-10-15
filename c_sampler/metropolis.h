#ifndef MCMC_LAB_METROPOLIS_H
#define MCMC_LAB_METROPOLIS_H

#include <stddef.h>

typedef struct {
    double mu;
    double sigma;
    double proposal_scale;
    unsigned int seed;
} NormalTargetConfig;

/* Draw n_samples from N(mu, sigma^2) with random-walk Metropolis-Hastings. */
int metropolis_normal(const NormalTargetConfig *config,
                      size_t n_samples,
                      double *out_samples);

/* Run independent chains in parallel (one OpenMP thread per chain). */
int metropolis_normal_chains(const NormalTargetConfig *base_config,
                             size_t n_chains,
                             size_t n_samples,
                             double *out_samples /* shape: n_chains * n_samples */);

#endif /* MCMC_LAB_METROPOLIS_H */