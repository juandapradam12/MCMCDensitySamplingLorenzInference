#include "metropolis.h"

#include <stdio.h>
#include <stdlib.h>

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
