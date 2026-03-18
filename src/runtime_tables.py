import numpy as np
import pandas as pd
import os

_DEFAULT_RESULTS_FOLDER = "results/"
method_map = {"DR-CF": "DR-KPT", "KPE": "KPT", "PE-linear": "PT-linear"}


def _result_path(scenario, ns, method, results_folder=_DEFAULT_RESULTS_FOLDER) -> str:
    """Single source of truth for CSV file paths."""
    return f"{results_folder}ns{ns}_scenario{scenario}_{method}.csv"


def load_results(results_folder=_DEFAULT_RESULTS_FOLDER) -> dict:
    d_results = {}

    # Null case (Scenario I)
    for method in ["PE-linear", "KPE", "DR-CF"]:
        for ns in np.arange(100, 1050, 50):
            fname = _result_path("I", ns, method, results_folder)
            if os.path.exists(fname):
                d_results[fname] = pd.read_csv(fname)
            else:
                print(f"Missing file: {fname}")

    # Alternative cases (Scenarios II–IV)
    for scenario in ["II", "III", "IV"]:
        for method in ["PE-linear", "KPE", "DR-CF"]:
            for ns in np.arange(100, 450, 50):
                fname = _result_path(scenario, ns, method, results_folder)
                if os.path.exists(fname):
                    d_results[fname] = pd.read_csv(fname)
                else:
                    print(f"Missing file: {fname}")

    return d_results


def build_scenario_table(scenario, sample_size_list, d_results, results_folder=_DEFAULT_RESULTS_FOLDER) -> pd.DataFrame:
    """d_results is now an explicit parameter (no hidden global)."""
    table_rows = []
    for method_key, method_latex in method_map.items():
        row = [method_latex]
        for ns in sample_size_list:
            fname = _result_path(scenario, ns, method_key, results_folder)
            if fname in d_results:
                df = d_results[fname]
                mean_time = df["time"].mean()
                std_time = df["time"].std()
                formatted = f"{mean_time:.3f} $\\pm$ {std_time:.3f}"
            else:
                formatted = "---"
            row.append(formatted)
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


if __name__ == "__main__":
    d_results = load_results()
    os.makedirs("tables", exist_ok=True)
    ns_list = list(np.arange(100, 450, 50))

    for scenario in ["I", "II", "III", "IV"]:
        df_scenario = build_scenario_table(scenario, ns_list, d_results)
        df_scenario.to_csv(f"tables/runtime_scenario_{scenario}.csv", index=False)

        latex_code = dataframe_to_latex_table(df_scenario, scenario)
        with open(f"tables/runtime_scenario_{scenario}.tex", "w") as f:
            f.write(latex_code)

        print(f"Saved Scenario {scenario} LaTeX table.")
