#include "metropolis.h"

#include <math.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double normal_log_density(double x, double mu, double sigma)
{
    const double z = (x - mu) / sigma;
    return -0.5 * z * z - log(sigma) - 0.5 * log(2.0 * M_PI);
}

static double urand(unsigned int *state)
{
    /* Portable xorshift32 -> U(0,1). */
    unsigned int x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x ? x : 0xA5A5A5A5u;
    return (*state >> 8) * (1.0 / 16777216.0);
}

int metropolis_normal(const NormalTargetConfig *config,
                      size_t n_samples,
                      double *out_samples)
{
    size_t i;
    unsigned int state;
    double current;
    double current_lp;
    size_t accepted = 0;

    if (!config || !out_samples || n_samples == 0 || config->sigma <= 0.0) {
        return -1;
    }

    state = config->seed ? config->seed : 1u;
    current = config->mu + config->sigma * (urand(&state) * 2.0 - 1.0);
    current_lp = normal_log_density(current, config->mu, config->sigma);

    for (i = 0; i < n_samples; ++i) {
        const double proposed =
            current + config->proposal_scale * (urand(&state) * 2.0 - 1.0);
        const double proposed_lp =
            normal_log_density(proposed, config->mu, config->sigma);
        const double log_alpha = proposed_lp - current_lp;

        if (log(urand(&state)) < log_alpha) {
            current = proposed;
            current_lp = proposed_lp;
            ++accepted;
        }
        out_samples[i] = current;
    }

    (void)accepted;
    return 0;
}

int metropolis_normal_chains(const NormalTargetConfig *base_config,
                             size_t n_chains,
                             size_t n_samples,
                             double *out_samples)
{
    size_t c;
    int status = 0;

    if (!base_config || !out_samples || n_chains == 0 || n_samples == 0) {
        return -1;
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (c = 0; c < n_chains; ++c) {
        NormalTargetConfig cfg = *base_config;
        int rc;
        cfg.seed = base_config->seed + (unsigned int)(c + 1u) * 9973u;
        rc = metropolis_normal(&cfg, n_samples, out_samples + c * n_samples);
        if (rc != 0) {
#if defined(_OPENMP)
#pragma omp atomic write
#endif
            status = -1;
        }
    }

    return status;
}
