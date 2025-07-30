import pandas as pd
import numpy as np
import json
import re
import argparse

# 銀行對應表
bank_mapping = {
    "first": "第一",
    "post": ["郵局", "郵政"],
    "chinatrust": "中信",
    "esun": "玉山",
    "union": "聯邦",
    "cooperative": "合作金庫",
    "shanghai": "上海",
    "taiwanbusiness": "台企",
    "bank": "銀行",
    "cathay": "國泰",
    "fubon": "富邦",
    "fareastern": "遠東",
    "land": "土地",
    "taishin": "台新",
    "taiwan": "台銀",
    "changhwa": "彰化",
    "huanan": "華南",
    "citi": "花旗",
    "mega": "兆豐",
    "yuanta": "元大",
    "jihsun": "日盛",
    "chartered": "渣打",
    "taichung": "台中",
    "sinopac": "永豐",
    "kingstown": "京城",
    "kauohsing": "高雄"
}

def translate_bank(bank_name):
    for eng, zh in bank_mapping.items():
        if isinstance(zh, list):
            if any(item in bank_name for item in zh):
                return eng
        else:
            if zh in bank_name:
                return eng
    return bank_name

def fix_mllm_result(val):
    try:
        if isinstance(val, str):
            val = json.loads(val)
        if not isinstance(val, dict):
            return {}

        result = {}
        for key, value in val.items():
            if "帳號" in key and isinstance(value, str):
                result[key] = value.replace("-", "")
            elif "銀行別" in key and isinstance(value, str):
                result[key] = translate_bank(value)
            else:
                result[key] = value
        return result
    except Exception as e:
        print(f"格式轉換失敗: {e}")
        return {}

def evaluate_results(path):
    df = pd.read_pickle(path)
    acc = {
        "戶名": 0,
        "銀行帳號": 0,
        "銀行別": 0
    }
    total = 0
    acc_keys = list(acc.keys())

    for idx in range(len(df)):
        gt = df["label"][idx]
        raw_pred = df["mllm_result"][idx]
        pred = fix_mllm_result(raw_pred)
        total += 1
        for key in acc_keys:
            if key in pred and gt[key] == pred[key]:
                acc[key] += 1

    for key in acc_keys:
        acc[key] /= total
    return acc

# 給模組化呼叫用的函數
def evaluate_covers(pred_name: str):
    pred_filename = pred_name if pred_name.endswith(".pkl") else pred_name + ".pkl"
    file_path = f"outputs/存摺封面/{pred_filename}"
    return evaluate_results(file_path)

# CLI 單獨執行
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", type=str, required=True, help="要評估的檔案名稱（不含副檔名）")
    args = parser.parse_args()

    result = evaluate_covers(args.pred_name)

    if result:
        for key, value in result.items():
            print(f"{key}: {value:.2%}")
        print("--------------")
        print(f"平均: {np.mean(list(result.values())):.2%}")
    else:
        print("無法計算準確度")

# python evaluation_存摺封面.py --pred_name=deepseek-vl2
