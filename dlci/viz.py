import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, probplot, gaussian_kde

# Colorblind-safe palette shared across all plots
_CB = {
    "hist": "#0072B2",
    "pdf": "#E69F00",
    "kde": "#999999",
    "qq": "#0D3B66",
    "errorbar": "#56B4E9",
    "nominal": "#4D4D4D",
}

_METHOD_LABELS = {
    "DR": "DR",
    "DR-CF": "DR-KPT",
    "KPE": "KPT",
    "PE-linear": "PT-linear",
    "NestedDR-CF": "Nested DR-KPT",
    "NestedDR": "Nested DR",
}

_METHOD_COLORS = {
    "PE-linear": "#E69F00",
    "KPE": "#0072B2",
    "DR-CF": "#009E73",
    "DR": "#56B4E9",
    "NestedDR-CF": "#009E73",
    "NestedDR": "#0072B2",
}

_MARKERS = ["^", "s", "v", "o", "D"]


def _fname(results_folder, ns, scenario, method, longitudinal):
    if longitudinal:
        return f"{results_folder}ns{ns}_longitudinal_scenario{scenario}_{method}.csv"
    return f"{results_folder}ns{ns}_scenario{scenario}_{method}.csv"


def _get(d, key):
    if key not in d:
        raise KeyError(
            f"Results not found: {key}\n"
            f"Run the experiment first, or check that results_folder is correct."
        )
    return d[key]


def plot_null_diagnostics(
    d,
    scenario="I",
    method="DR-CF",
    ns=350,
    results_folder="results/",
    ns_list=None,
    confidence_level=0.05,
    save_path="plots/null_diagnostics.pdf",
    longitudinal=False,
):
    """Three-panel null diagnostic: (A) histogram+KDE vs N(0,1), (B) QQ-plot, (C) FPR vs n.

    Parameters
    ----------
    d               : dict returned by load_results or load_longitudinal_results
    scenario        : scenario ID, e.g. "I"
    method          : method key, e.g. "DR-CF" or "NestedDR-CF"
    ns              : sample size used for panels A and B
    results_folder  : path prefix for CSV files
    ns_list         : sample sizes for the FPR curve (panel C)
    confidence_level: nominal level α
    save_path       : output file path
    longitudinal    : use longitudinal CSV naming if True
    """
    if ns_list is None:
        ns_list = list(
            np.arange(100, 550, 50) if longitudinal else np.arange(100, 1050, 50)
        )

    plt.rcParams["figure.figsize"] = (18, 5)
    plt.rcParams["axes.grid"] = True
    plt.rc("axes", labelsize=13)
    plt.rc("xtick", labelsize=11)
    plt.rc("ytick", labelsize=11)

    label = _METHOD_LABELS.get(method, method)
    stat_values = _get(d, _fname(results_folder, ns, scenario, method, longitudinal))[
        "stat"
    ]
    x_axis = np.linspace(-3, 3, 500)

    # (A) Histogram + KDE + N(0,1)
    plt.subplot(1, 3, 1)
    plt.hist(
        stat_values,
        bins=25,
        density=True,
        color=_CB["hist"],
        alpha=0.5,
        label=f"Draws from {label}",
        edgecolor="black",
    )
    plt.plot(
        x_axis,
        gaussian_kde(stat_values)(x_axis),
        color=_CB["kde"],
        linestyle="--",
        label="KDE",
    )
    plt.plot(x_axis, norm.pdf(x_axis), color=_CB["pdf"], linewidth=2, label="N(0,1)")
    plt.xlabel(label)
    plt.ylabel("Density")
    plt.title("(A)")
    plt.legend(loc="upper left")

    # (B) QQ-plot
    plt.subplot(1, 3, 2)
    osm, osr = probplot(stat_values, dist="norm")[0]
    plt.plot(osm, osr, marker="o", linestyle="", color=_CB["qq"], markersize=3)
    plt.plot(osm, osm, color="black", linestyle="--")
    plt.title("(B)")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Ordered values")

    # (C) FPR vs sample size
    plt.subplot(1, 3, 3)
    rates = np.array(
        [
            (
                _get(d, _fname(results_folder, n, scenario, method, longitudinal))[
                    "p_value"
                ]
                < confidence_level
            ).mean()
            for n in ns_list
        ]
    )
    n_rep = len(
        _get(d, _fname(results_folder, ns_list[0], scenario, method, longitudinal))[
            "p_value"
        ]
    )
    varhat = rates * (1 - rates) / n_rep
    plt.errorbar(
        ns_list,
        rates,
        yerr=1.96 * np.sqrt(varhat),
        capsize=4,
        marker="^",
        markersize=8,
        color=_CB["errorbar"],
        linestyle="--",
        label=label,
    )
    plt.axhline(
        confidence_level, color=_CB["nominal"], linestyle="--", label="Nominal level"
    )
    plt.title("(C)")
    plt.xlabel("Sample size")
    plt.ylabel("False positive rate")
    plt.legend(loc="upper right")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_power(
    d,
    scenario_list=("II", "III", "IV"),
    methods=("DR-CF",),
    ns_list=None,
    results_folder="results/",
    confidence_level=0.05,
    save_path="plots/power.pdf",
    longitudinal=False,
):
    """Power curves (TPR vs sample size) for one or more scenarios and methods.

    Parameters
    ----------
    d               : dict returned by load_results or load_longitudinal_results
    scenario_list   : scenarios to plot (one subplot each)
    methods         : method keys to overlay on each subplot
    ns_list         : sample sizes for the x-axis
    results_folder  : path prefix for CSV files
    confidence_level: rejection threshold α
    save_path       : output file path
    longitudinal    : use longitudinal CSV naming if True
    """
    if ns_list is None:
        ns_list = list(np.arange(100, 450, 50))

    plt.rcParams["figure.figsize"] = (16, 4)
    plt.rc("legend", fontsize=12)
    plt.rc("axes", labelsize=15)
    plt.rc("xtick", labelsize=15)
    plt.rc("ytick", labelsize=15)
    plt.rcParams["axes.grid"] = True

    scenario_titles = {
        "II": "(II) Mean shift",
        "III": "(III) Mixture",
        "IV": "(IV) Feedback-only" if longitudinal else "(IV)",
    }
    ns_array = np.array(ns_list)

    fig, axs = plt.subplots(1, len(scenario_list), constrained_layout=True)
    if len(scenario_list) == 1:
        axs = [axs]

    for col, scenario in enumerate(scenario_list):
        ax = axs[col]
        ax.set_title(scenario_titles.get(scenario, scenario))
        ax.set_ylim((-0.05, 1.05))
        ax.set_xlabel("Sample size")
        ax.set_xticks(ns_list)
        ax.grid(True, linestyle="--", alpha=0.6)
        if col == 0:
            ax.set_ylabel("True positive rate")

        for i, method in enumerate(methods):
            rates = np.array(
                [
                    (
                        _get(
                            d,
                            _fname(results_folder, ns, scenario, method, longitudinal),
                        )["p_value"]
                        < confidence_level
                    ).mean()
                    for ns in ns_list
                ]
            )
            n_rep = len(
                _get(
                    d,
                    _fname(results_folder, ns_list[0], scenario, method, longitudinal),
                )["p_value"]
            )
            varhat = rates * (1 - rates) / n_rep
            ax.errorbar(
                ns_array,
                rates,
                yerr=1.96 * np.sqrt(varhat),
                capsize=4,
                marker=_MARKERS[i % len(_MARKERS)],
                linestyle="--",
                linewidth=1.5,
                markersize=8,
                label=_METHOD_LABELS.get(method, method),
                color=_METHOD_COLORS.get(method, "#333333"),
            )
        ax.axhline(confidence_level, color="black", linestyle=":", linewidth=1)
        ax.legend(loc="lower right" if col > 0 else "lower left")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
