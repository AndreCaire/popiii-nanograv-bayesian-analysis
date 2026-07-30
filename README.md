# Population III Gravitational-Wave Background: Bayesian Analysis with NANOGrav

Bayesian analysis of a phenomenological Population III compact-binary merger model using the NANOGrav 15-year free-spectrum data and the `ceffyl` factorized-likelihood framework.

This project was developed during a visiting research period at the University of Birmingham.

## Scientific workflow

1. Define a Population III merger-rate model as a function of redshift and chirp mass.
2. Calculate the gravitational-wave background contribution over a grid of `alpha`, `beta`, and `gamma`.
3. Load the precomputed physical grid using multidimensional interpolation.
4. Convert the predicted characteristic strain into a power spectral density.
5. Evaluate the model against the NANOGrav 15-year free-spectrum likelihood using `ceffyl`.
6. Sample the posterior distribution with `PTMCMCSampler`.

## Inferred parameters

- `log10_R0`: logarithm of the merger-rate normalization
- `alpha`: redshift-evolution parameter
- `beta`: high-redshift cutoff parameter
- `gamma`: mass-distribution slope

## Repository structure

- `data/real_physics_grid.txt`: precomputed physical integral grid
- `notebooks/popIII_nanograv_bayesian_analysis.ipynb`: research notebook
- `results/chain_1.txt`: stored MCMC chain
- `results/pars.txt`: sampled parameter names

## Main software

- Python
- NumPy
- SciPy
- Matplotlib
- corner
- ceffyl 1.41.2
- enterprise-pulsar
- PTMCMCSampler

## Reproducibility

The notebook follows a sequential workflow. Later cells depend on objects and functions initialized in earlier cells.

The included chain contains 5,001 stored states from a 50,000-step MCMC run with thinning by a factor of 10.

Additional convergence diagnostics should be performed before drawing definitive astrophysical conclusions.

## Author

André Caire  
M.Sc. candidate in Asymptotically Safe Quantum Gravity at São Paulo State University (UNESP)
