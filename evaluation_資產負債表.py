import pandas as pd
import numpy as np
import argparse
import re
import sys
import os

# ----------------------------
# 將 mllm_result 字串轉為 dict
# ----------------------------
def fix_mllm_result(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            val = val.replace("\\n", " ").replace("\r", " ").replace("\\t", " ").strip()
            kv_pairs = re.findall(r'"?([^":]+)"?\s*:\s*"?([^\{\}\"\':,]+)?"?', val)
            result = {}
            for k, v in kv_pairs:
                v = v.replace(",", "")
                if v.isdigit():
                    result[k] = int(v)
                else:
                    result[k] = v
            return result
        except Exception as e:
            print(f"[fix_mllm_result] 格式轉換失敗：{e}")
            return {}
    return {}

# ----------------------------
# 主準確率計算函數
# ----------------------------
def evaluate_results(path):
    df = pd.read_pickle(path)
    df["mllm_result"] = df["mllm_result"].apply(fix_mllm_result)

    acc = {
        "營利事業名稱": 0,
        "日期": 0,
        "1100流動資產": 0,
        "1111現金": 0,
        "1112銀行存款": 0,
        "1113約當現金": 0,
        "1151透過損益按公允價值衡量之金融資產-流動(附註三)": 0,
        "1158透過其他綜合損益按公允價值衡量之金融資產－流動(附註三)": 0,
        "1161按攤銷後成本衡量之金融資產-流動(附註三)": 0,
        "1121應收票據": 0,
        "1131應收帳款": 0,
        "1130存貨": 0,
        "1192業主(股東)往來": 0,
        "1200非流動資產": 0,
        "1612透過損益按公允價值衡量之金融資產-非流動(附註三)": 0,
        "1615透過其他綜合損益按公允價值衡量之金融資產-非流動(附註三)": 0,
        "1622按攤銷後成本衡量之金融資產-非流動(附註三)": 0,
        "1400不動產、廠房及設備(固定資產)": 0,
        "2100流動負債": 0,
        "2111銀行透支": 0,
        "2112銀行借款": 0,
        "2113應付短期票券": 0,
        "2120應付票據": 0,
        "2121應付帳款": 0,
        "2192業主(股東)往來": 0,
        "2200非流動負債": 0,
        "2210應付公司債": 0,
        "2220長期借款": 0,
        "2000負債總額": 0,
        "3100資本或股本(實收)": 0,
        "3300資本公積": 0,
        "3400保留盈餘": 0,
        "3000權益總額": 0
    }

    total = len(df)
    acc_keys = list(acc.keys())

    for idx in range(total):
        gt = df['label'][idx]
        pred = df['mllm_result'][idx]
        for key in acc_keys:
            if isinstance(pred, dict) and key in pred and key in gt:
                if gt[key] == pred[key]:
                    acc[key] += 1

    for key in acc_keys:
        acc[key] /= total if total > 0 else 1

    return acc

# 模組化呼叫函數
def evaluate_balance(pred_name: str):
    pred_filename = pred_name if pred_name.endswith(".pkl") else pred_name + ".pkl"
    file_path = os.path.join("outputs", "資產負債表", pred_filename)
    return evaluate_results(file_path)

# ----------------------------
# 主程式（可 CLI 執行）
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", type=str, required=True, help="output 資料夾中的檔名，不含副檔名，例如：modelA")
    args = parser.parse_args()

    pred_filename = args.pred_name if args.pred_name.endswith(".pkl") else args.pred_name + ".pkl"
    file_path = os.path.join("outputs", "資產負債表", pred_filename)

    if not os.path.exists(file_path):
        print(f"找不到檔案：{file_path}")
        sys.exit(1)

    acc = evaluate_results(file_path)

    for k, v in acc.items():
        print(f"{k}: {v:.2%}")
    print("----")
    print(f"平均: {np.mean(list(acc.values())):.2%}")

# python evaluation_資產負債表.py --pred_name=deepseek_vl2