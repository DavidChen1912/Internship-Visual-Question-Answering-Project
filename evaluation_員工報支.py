import pandas as pd
import numpy as np
import json
import os
import re
import argparse

def fix_mllm_result(val):
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [val]
    if isinstance(val, str):
        try:
            val = val.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
            if not val.startswith("["):
                val = "[" + val
            if not val.endswith("]"):
                val += "]"
            parsed = json.loads(val)
            if not isinstance(parsed, list):
                return []
            cleaned_list = []
            for entry in parsed:
                if not isinstance(entry, dict):
                    continue
                new_entry = {}
                for k, v in entry.items():
                    if k in ["賣方統編", "發票號碼"]:
                        v = v.replace("-", "")
                    if k == "憑證日期" and isinstance(v, str) and len(v) >= 10:
                        v = v[:10]
                    new_entry[k] = v
                cleaned_list.append(new_entry)
            return cleaned_list
        except Exception as e:
            print(f"[fix_mllm_result] 格式轉換失敗: {e}")
            return []
    return []

def evaluate_results(path):
    df = pd.read_pickle(path)
    df["mllm_result"] = df["mllm_result"].apply(fix_mllm_result)
    eval_keys = ["憑證類別", "賣方統編", "發票號碼", "憑證日期", "金額", "稅額", "銷售額"]
    acc = {k: 0 for k in eval_keys}
    total = 0
    for idx, row in df.iterrows():
        gt_list = row["label"]
        pred_list = row["mllm_result"]
        if not isinstance(gt_list, list) or not isinstance(pred_list, list):
            continue
        if len(gt_list) == 0:
            continue
        total += 1
        gt = gt_list[0]
        pred = pred_list[0] if len(pred_list) > 0 else {}
        for k in eval_keys:
            if k in pred and k in gt and pred[k] == gt[k]:
                acc[k] += 1
    for k in acc:
        acc[k] /= total if total > 0 else 1
    return acc

# 給模組化呼叫用的函數
def evaluate_expenses(pred_name: str):
    pred_filename = pred_name if pred_name.endswith(".pkl") else pred_name + ".pkl"
    file_path = os.path.join("outputs", "員工報支", pred_filename)
    return evaluate_results(file_path)

# 可單獨執行用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", type=str, required=True, help="要評估的檔案名稱（不含副檔名）")
    args = parser.parse_args()
    results = evaluate_expenses(args.pred_name)
    for k, v in results.items():
        print(f"{k}: {v:.2%}")
    print("--------------")
    print(f"平均: {np.mean(list(results.values())):.2%}")

# python evaluation_員工報支.py --pred_name=deepseek_vl2