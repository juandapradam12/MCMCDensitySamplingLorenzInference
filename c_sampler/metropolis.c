#include "metropolis.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(| : status)
#endif
    for (c = 0; c < n_chains; ++c) {
        NormalTargetConfig cfg = *base_config;
        cfg.seed = base_config->seed + (unsigned int)(c + 1u) * 9973u;
        if (metropolis_normal(&cfg, n_samples, out_samples + c * n_samples) != 0) {
            status = -1;
        }
    }

    return status;
}

static void write_chain(const char *path, const double *samples, size_t n)
{
    size_t i;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        perror(path);
        exit(1);
    }
    for (i = 0; i < n; ++i) {
        fprintf(fp, "%.8f\n", samples[i]);
    }
    fclose(fp);
}

int main(int argc, char **argv)
{
    NormalTargetConfig config;
    size_t n_chains = 8;
    size_t n_samples = 2000;
    size_t burnin;
    size_t c;
    double *samples;
    char path[64];

    config.mu = 0.0;
    config.sigma = 1.0;
    config.proposal_scale = 1.0;
    config.seed = 42u;

    if (argc > 1) {
        n_samples = (size_t)atoi(argv[1]);
    }
    if (argc > 2) {
        n_chains = (size_t)atoi(argv[2]);
    }
    burnin = n_samples / 4;
    if (burnin >= n_samples) {
        burnin = 0;
    }

    samples = (double *)malloc(sizeof(double) * n_chains * n_samples);
    if (!samples) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    if (metropolis_normal_chains(&config, n_chains, n_samples, samples) != 0) {
        fprintf(stderr, "sampler failed\n");
        free(samples);
        return 1;
    }

    for (c = 0; c < n_chains; ++c) {
        const double *chain = samples + c * n_samples;
        double mean = 0.0;
        size_t i;
        size_t count = 0;

        snprintf(path, sizeof(path), "sample_%zu.dat", c);
        /* Files are written relative to the process cwd (typically c_sampler/bin). */
        write_chain(path, chain, n_samples);

        for (i = burnin; i < n_samples; ++i) {
            mean += chain[i];
            ++count;
        }
        mean /= (double)count;
        printf("chain %zu mean (post-burnin)=%.4f -> %s\n", c, mean, path);
    }

    free(samples);
    return 0;
}