"""Diagnostics for the Population III toy-model MCMC chain.

This script loads the PTMCMC chain, removes the first 25% as burn-in,
creates trace/corner/autocorrelation plots, and prints posterior summaries.

Gelman-Rubin R-hat is intentionally not computed because a valid R-hat
analysis requires multiple independent chains.
"""

from __future__ import annotations

from pathlib import Path

import corner
import matplotlib.pyplot as plt
import numpy as np


CHAIN_DIR = Path("chains_toy_final")
CHAIN_FILE = CHAIN_DIR / "chain_1.txt"
PARAMS_FILE = CHAIN_DIR / "pars.txt"

BURN_IN_FRACTION = 0.25
MAX_AUTOCORR_LAG = 1000

TRACEPLOT_FILE = Path("diagnostics_traceplot_corrected.png")
CORNER_FILE = Path("diagnostics_corner_corrected.png")
AUTOCORR_FILE = Path("diagnostics_autocorr_corrected.png")


def load_chain() -> tuple[np.ndarray, list[str]]:
    """Load the raw PTMCMC chain and parameter names."""
    if not CHAIN_FILE.exists():
        raise FileNotFoundError(f"Chain file not found: {CHAIN_FILE}")

    if not PARAMS_FILE.exists():
        raise FileNotFoundError(f"Parameter file not found: {PARAMS_FILE}")

    chain = np.loadtxt(CHAIN_FILE)
    parameter_names = np.atleast_1d(
        np.loadtxt(PARAMS_FILE, dtype=str)
    ).tolist()

    if chain.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional chain, received shape {chain.shape}."
        )

    if chain.shape[1] < len(parameter_names):
        raise ValueError(
            "The chain has fewer columns than the number of listed parameters."
        )

    return chain, parameter_names


def remove_burn_in(
    chain: np.ndarray,
    number_of_parameters: int,
) -> tuple[np.ndarray, int]:
    """Remove burn-in and retain only parameter columns."""
    burn_in_samples = int(BURN_IN_FRACTION * chain.shape[0])
    posterior_samples = chain[burn_in_samples:, :number_of_parameters]

    if posterior_samples.size == 0:
        raise ValueError("No samples remain after burn-in removal.")

    return posterior_samples, burn_in_samples


def autocorrelation_fft(values: np.ndarray) -> np.ndarray:
    """Return the normalized autocorrelation using an FFT-based method."""
    centered = np.asarray(values, dtype=float) - np.mean(values)
    sample_count = centered.size

    if sample_count < 2:
        raise ValueError("At least two samples are required.")

    fft_size = 1 << (2 * sample_count - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocovariance = np.fft.irfft(
        spectrum * np.conjugate(spectrum),
        n=fft_size,
    )[:sample_count]

    autocovariance /= np.arange(sample_count, 0, -1)

    if autocovariance[0] == 0:
        return np.ones(sample_count)

    return autocovariance / autocovariance[0]


def integrated_autocorrelation_time(
    autocorrelation: np.ndarray,
) -> float:
    """Estimate integrated autocorrelation time from the first positive run."""
    positive_lags = autocorrelation[1:]
    non_positive = np.flatnonzero(positive_lags <= 0)
    stop = int(non_positive[0]) if non_positive.size else len(positive_lags)
    tau = 1.0 + 2.0 * np.sum(positive_lags[:stop])
    return max(float(tau), 1.0)


def create_traceplot(
    samples: np.ndarray,
    parameter_names: list[str],
) -> None:
    """Create and save trace plots for all parameters."""
    figure, axes = plt.subplots(
        len(parameter_names),
        1,
        figsize=(11, 3.2 * len(parameter_names)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)

    for index, axis in enumerate(axes):
        values = samples[:, index]
        median = np.median(values)

        axis.plot(values, alpha=0.55, linewidth=0.45)
        axis.axhline(
            median,
            linestyle="--",
            linewidth=1.2,
            label=f"Median: {median:.3f}",
        )
        axis.set_ylabel(parameter_names[index])
        axis.grid(alpha=0.25)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Post-burn-in sample")
    figure.suptitle(
        "Population III toy model: trace plots "
        f"({BURN_IN_FRACTION:.0%} burn-in removed)"
    )
    figure.tight_layout()
    figure.savefig(TRACEPLOT_FILE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_corner_plot(
    samples: np.ndarray,
    parameter_names: list[str],
) -> None:
    """Create and save the posterior corner plot."""
    figure = corner.corner(
        samples,
        labels=parameter_names,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 11},
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
    )
    figure.savefig(CORNER_FILE, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_autocorrelation_plot(
    samples: np.ndarray,
    parameter_names: list[str],
) -> list[tuple[float, float]]:
    """Create autocorrelation plots and return tau/ESS estimates."""
    figure, axes = plt.subplots(
        len(parameter_names),
        1,
        figsize=(9, 2.8 * len(parameter_names)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    diagnostics: list[tuple[float, float]] = []

    for index, axis in enumerate(axes):
        autocorrelation = autocorrelation_fft(samples[:, index])
        maximum_lag = min(MAX_AUTOCORR_LAG, len(autocorrelation) - 1)

        tau = integrated_autocorrelation_time(autocorrelation)
        effective_sample_size = samples.shape[0] / tau
        diagnostics.append((tau, effective_sample_size))

        axis.plot(
            np.arange(maximum_lag + 1),
            autocorrelation[: maximum_lag + 1],
            linewidth=1.0,
        )
        axis.axhline(0.0, linestyle="--", linewidth=0.8)
        axis.set_ylabel(parameter_names[index])
        axis.grid(alpha=0.25)
        axis.text(
            0.99,
            0.88,
            f"tau ≈ {tau:.1f}\nESS ≈ {effective_sample_size:.0f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
        )

    axes[-1].set_xlabel("Lag")
    figure.suptitle("Population III toy model: autocorrelation")
    figure.tight_layout()
    figure.savefig(AUTOCORR_FILE, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return diagnostics


def print_summary(
    raw_chain: np.ndarray,
    samples: np.ndarray,
    burn_in_samples: int,
    parameter_names: list[str],
    diagnostics: list[tuple[float, float]],
) -> None:
    """Print chain dimensions, quantiles, and diagnostics."""
    print("=" * 72)
    print("Population III toy-model MCMC diagnostics")
    print("=" * 72)
    print(f"Raw chain shape:        {raw_chain.shape}")
    print(f"Burn-in removed:        {burn_in_samples} samples")
    print(f"Posterior sample shape: {samples.shape}")
    print()

    for index, name in enumerate(parameter_names):
        values = samples[:, index]
        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        lower = q50 - q16
        upper = q84 - q50
        tau, effective_sample_size = diagnostics[index]

        print(
            f"{name:24s} {q50: .5f} "
            f"(+{upper:.5f} / -{lower:.5f}) | "
            f"tau ≈ {tau:.2f} | ESS ≈ {effective_sample_size:.0f}"
        )

    print()
    print(f"Saved: {TRACEPLOT_FILE}")
    print(f"Saved: {CORNER_FILE}")
    print(f"Saved: {AUTOCORR_FILE}")
    print()
    print(
        "R-hat was not computed because a valid Gelman-Rubin analysis "
        "requires multiple independent chains."
    )


def main() -> None:
    raw_chain, parameter_names = load_chain()
    samples, burn_in_samples = remove_burn_in(
        raw_chain,
        number_of_parameters=len(parameter_names),
    )

    create_traceplot(samples, parameter_names)
    create_corner_plot(samples, parameter_names)
    diagnostics = create_autocorrelation_plot(samples, parameter_names)

    print_summary(
        raw_chain,
        samples,
        burn_in_samples,
        parameter_names,
        diagnostics,
    )


if __name__ == "__main__":
    main()