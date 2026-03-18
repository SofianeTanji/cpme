import glob
import re
import numpy as np
import pandas as pd
import os

_DEFAULT_RESULTS_FOLDER = "results/"
method_map = {"DR-CF": "DR-KPT", "KPE": "KPT", "PE-linear": "PT-linear"}
longitudinal_method_map = {"NestedDR-CF": "Nested DR-KPT"}


def _infer_methods(d_results, scenario, longitudinal=False) -> list:
    if longitudinal:
        pattern = rf"ns\d+_longitudinal_scenario{re.escape(scenario)}_(.+)\.csv$"
    else:
        pattern = rf"ns\d+_scenario{re.escape(scenario)}_(.+)\.csv$"
    methods = set()
    for fname in d_results:
        m = re.search(pattern, fname)
        if m:
            methods.add(m.group(1))
    return sorted(methods)


def _result_path(scenario, ns, method, results_folder=_DEFAULT_RESULTS_FOLDER) -> str:
    # Single source of truth for CSV naming — notebooks and tables must use this.
    return f"{results_folder}ns{ns}_scenario{scenario}_{method}.csv"


def load_results(results_folder=_DEFAULT_RESULTS_FOLDER) -> dict:
    pattern = os.path.join(results_folder, "ns*_scenario*.csv")
    files = [f for f in glob.glob(pattern) if "_longitudinal_" not in f]
    return {f: pd.read_csv(f) for f in sorted(files)}


def build_scenario_table(
    scenario, sample_size_list, d_results, results_folder=_DEFAULT_RESULTS_FOLDER
) -> pd.DataFrame:
    table_rows = []
    for method in _infer_methods(d_results, scenario, longitudinal=False):
        row = [method_map.get(method, method)]
        for ns in sample_size_list:
            fname = _result_path(scenario, ns, method, results_folder)
            if fname in d_results:
                df = d_results[fname]
                formatted = f"{df['time'].mean():.3f} $\\pm$ {df['time'].std():.3f}"
            else:
                formatted = "---"
            row.append(formatted)
        table_rows.append(row)
    columns = ["Method"] + [str(ns) for ns in sample_size_list]
    return pd.DataFrame(table_rows, columns=columns)


def _longitudinal_result_path(
    scenario, ns, method, results_folder=_DEFAULT_RESULTS_FOLDER
) -> str:
    # Single source of truth for longitudinal CSV naming.
    return f"{results_folder}ns{ns}_longitudinal_scenario{scenario}_{method}.csv"


def load_longitudinal_results(results_folder=_DEFAULT_RESULTS_FOLDER) -> dict:
    pattern = os.path.join(results_folder, "ns*_longitudinal_scenario*.csv")
    return {f: pd.read_csv(f) for f in sorted(glob.glob(pattern))}


def build_longitudinal_scenario_table(
    scenario,
    sample_size_list,
    d_results,
    alpha=0.05,
    results_folder=_DEFAULT_RESULTS_FOLDER,
) -> pd.DataFrame:
    """Rejection-rate table (mean ± 1.96 SE) for longitudinal scenarios."""
    table_rows = []
    for method in _infer_methods(d_results, scenario, longitudinal=True):
        row = [longitudinal_method_map.get(method, method)]
        for ns in sample_size_list:
            fname = _longitudinal_result_path(scenario, ns, method, results_folder)
            if fname in d_results:
                df = d_results[fname]
                rate = (df["p_value"] < alpha).mean()
                se = np.sqrt(rate * (1 - rate) / len(df))
                row.append(f"{rate:.2f} $\\pm$ {1.96 * se:.2f}")
            else:
                row.append("---")
        table_rows.append(row)
    columns = ["Method"] + [str(ns) for ns in sample_size_list]
    return pd.DataFrame(table_rows, columns=columns)


def dataframe_to_latex_table(df, scenario):
    header = (
        f"\\begin{{table}}[h]\n"
        f"\\centering\n"
        f"\\caption{{Average runtime (in seconds) for Scenario {scenario}. Values are reported as mean $\\pm$ std over 100 runs.}}\n"
        f"\\label{{tab:runtime_scenario_{scenario.lower()}}}\n"
        f"\\resizebox{{\\textwidth}}{{!}}{{\n"
        f"\\begin{{tabular}}{{l" + "c" * (df.shape[1] - 1) + "}\n"
        f"\\toprule\n"
        + " & ".join(f"\\textbf{{{col}}}" for col in df.columns)
        + " \\\\\n"
        f"\\midrule\n"
    )
    body = ""
    for _, row in df.iterrows():
        row_str = " & ".join(str(cell) for cell in row) + " \\\\\n"
        body += row_str
    footer = "\\bottomrule\n\\end{tabular}}\n\\end{table}"
    return header + body + footer


def main():
    os.makedirs("tables", exist_ok=True)

    # Non-nested runtime tables
    d_results = load_results()
    ns_list = list(np.arange(100, 450, 50))
    for scenario in ["I", "II", "III", "IV"]:
        df_scenario = build_scenario_table(scenario, ns_list, d_results)
        df_scenario.to_csv(f"tables/runtime_scenario_{scenario}.csv", index=False)
        latex_code = dataframe_to_latex_table(df_scenario, scenario)
        with open(f"tables/runtime_scenario_{scenario}.tex", "w") as f:
            f.write(latex_code)
        print(f"Saved Scenario {scenario} LaTeX table.")

    # Longitudinal rejection-rate tables
    d_long = load_longitudinal_results()
    for scenario in ["I", "II", "III", "IV"]:
        ns_list_long = list(
            np.arange(100, 550, 50) if scenario == "I" else np.arange(100, 450, 50)
        )
        df_long = build_longitudinal_scenario_table(scenario, ns_list_long, d_long)
        df_long.to_csv(f"tables/longitudinal_scenario_{scenario}.csv", index=False)
        print(f"Saved longitudinal Scenario {scenario} table.")


if __name__ == "__main__":
    main()
