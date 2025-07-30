import pandas as pd
import numpy as np
import argparse
import os
import json
import re

# ----------------------------
# 字串轉 dict 並清理格式
# ----------------------------
def ensure_dict(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            val = val.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
            val = re.sub(r"%(?=[,\}\]])", "", val)
            val = re.sub(r"(\d+)\.\d+", r"\1", val)
            val = re.sub(r'"\s*(-?\d+,?\d+)(?=[,\}\]])', lambda m: '"' + m.group(1).replace(",", ""), val)
            return json.loads(val)
        except Exception:
            return {}
    return {}

# ----------------------------
# 準確率評估函數（主邏輯）
# ----------------------------
def evaluate_results(pred_path):
    df = pd.read_pickle(pred_path)

    acc = {
        "01營業收入總額": 0,
        "04營業收入淨額": 0,
        "05營業成本": 0,
        "08營業費用及損失總額": 0,
        "35投資收益": 0,
        "36依所得稅法第42條規定取得之股利或盈餘": 0,
        "38利息收入": 0,
        "39租賃收入": 0,
        "40處分資產利益": 0,
        "43兌換盈餘": 0,
        "44其他收入": 0,
        "46利息支出": 0,
        "47投資損失": 0,
        "48處分資產損失": 0,
        "51兌換損失": 0,
        "53全併所得額": 0,
        "59課稅所得額": 0,
        "60本年度應納稅額": 0
    }

    total = 0
    acc_keys = list(acc.keys())

    for idx in range(len(df)):
        gt = ensure_dict(df['label'][idx])
        pred = ensure_dict(df['mllm_result'][idx])
        total += 1

        for key in acc_keys:
            if key in pred and key in gt and gt[key] == pred[key]:
                acc[key] += 1

    for key in acc_keys:
        acc[key] /= total if total > 0 else 1

    return acc

# 給模組化呼叫用的函數
def evaluate_income(pred_name: str):
    pred_path = os.path.join("outputs", "損益表", f"{pred_name}.pkl")
    return evaluate_results(pred_path)

# ----------------------------
# 主程式入口：CLI 執行
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", type=str, required=True, help="模型推理結果的檔案名稱（不含副檔名），會自動讀取 outputs/損益表/{pred_name}.pkl")
    args = parser.parse_args()

    pred_path = os.path.join("outputs", "損益表", f"{args.pred_name}.pkl")

    if not os.path.exists(pred_path):
        print(f" 找不到檔案：{pred_path}")
        exit(1)

    accuracy_results = evaluate_results(pred_path)

    # 顯示結果
    if accuracy_results:
        for key, value in accuracy_results.items():
            print(f"{key}: {value:.2%}")
        overall = np.mean([value for value in accuracy_results.values()])
        print("----------------------")
        print(f"平均準確率: {overall:.2%}")
    else:
        print(" 無法計算準確度")

# python evaluation_損益表.py --pred_name=deepseek_vl2

