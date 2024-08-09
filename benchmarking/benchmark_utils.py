from typing import Any, Dict
import pandas as pd


def compare_benchmarks(benchmarks: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    series_dict = {k: pd.Series(v.values()) for k, v in benchmarks.items()}
    series_dict["kernel_path"] = pd.Series(benchmarks[list(benchmarks.keys())[0]].keys())
    series_dict["kernel"] = pd.Series([k.split(".")[-1] for k in series_dict["kernel_path"]])
    for p in  series_dict["kernel_path"]:
        print(p)
    raise ValueError("test")
    df = pd.DataFrame()
    for k, v in series_dict.items():
        df[k] = v
    columns = [c for c in df.columns if not "kernel" in c]
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            # calculate the difference between the two columns
            diff_col_name = f"{columns[i]}-{columns[j]}"
            df[diff_col_name] = df[columns[i]] - df[columns[j]]
    df.sort_values(by = 'kernel_path', inplace = True)
    columns = [c for c in df.columns if not "kernel" in c]
    columns = ["kernel", "kernel_path"] + columns
    df = df[columns]
    df.set_index("kernel", inplace=True)
    return df
