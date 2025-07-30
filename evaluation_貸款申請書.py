import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import re

# ----------------------------
# 修正 mllm_result 結構
# ----------------------------
def fix_mllm_result(val):
    expected_keys = [
        "申請金額", "借款期間", "償還方式", "貸款用途", "申請人姓名", "申請人身分證字號",
        "申請人生日", "申請人婚姻狀況", "申請人子女人數", "申請人住宅地址", "申請人年資",
        "申請人公司統編", "申請人年收入", "申請人行動電話", "申請人住宅電話", "申請人公司電話",
        "保證人姓名", "保證人身分證字號", "保證人生日"
    ]
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            result = json.loads(val)
            for key in expected_keys:
                if key not in result:
                    result[key] = ""
            return result
        except Exception:
            return {}
    return {}

# ----------------------------
# 主準確率計算函數
# ----------------------------
def evaluate_results(path):
    df = pd.read_pickle(path)
    df["mllm_result"] = df["mllm_result"].apply(fix_mllm_result)

    acc = {
        "申請金額": 0, "借款期間": 0, "償還方式": 0, "貸款用途": 0, "申請人姓名": 0,
        "申請人身分證字號": 0, "申請人生日": 0, "申請人婚姻狀況": 0, "申請人子女人數": 0,
        "申請人住宅地址": 0, "申請人年資": 0, "申請人公司統編": 0, "申請人年收入": 0,
        "申請人行動電話": 0, "申請人住宅電話": 0, "申請人公司電話": 0,
        "保證人姓名": 0, "保證人身分證字號": 0, "保證人生日": 0
    }

    total = len(df)
    acc_keys = list(acc.keys())

    for idx in range(total):
        gt = df['label'][idx]
        pred = df['mllm_result'][idx]
        for key in acc_keys:
            try:
                if key == "償還方式":
                    if "本息攤還" in gt[key] and "本息攤還" in pred[key]:
                        acc[key] += 0.5
                        index = gt[key].find('(')
                        if index != -1 and gt[key][index:] == pred[key][index:]:
                            acc[key] += 0.5
                    elif gt[key] == pred[key]:
                        acc[key] += 1
                elif key == "申請人住宅地址":
                    if "同身分證戶籍地址" in gt[key] and "同身分證戶籍地址" in pred[key]:
                        acc[key] += 1
                    elif gt[key] == pred[key]:
                        acc[key] += 1
                elif key == "貸款用途":
                    gt_items = gt[key].split("；")
                    matched_count = 0
                    for item in gt_items:
                        index = item.find("(")
                        if item.strip() in pred[key]:
                            matched_count += 1
                        elif index != -1 and item[index:] in pred[key]:
                            matched_count += 0.5
                    if len(gt_items) > 0:
                        acc[key] += matched_count / len(gt_items)
                elif key in ["申請金額", "借款期間", "申請人子女人數", "申請人年資", "申請人年收入"]:
                    pred_val = re.sub(r"[^\d]", "", pred[key])
                    if gt[key] == pred_val:
                        acc[key] += 1
                else:
                    if gt[key] == pred[key]:
                        acc[key] += 1
            except:
                pass

    for key in acc_keys:
        acc[key] /= total if total > 0 else 1

    return acc

# 模組化函數（給其他腳本匯入使用）
def evaluate_loan(pred_name: str):
    file_path = os.path.join("outputs", "貸款申請書", f"{pred_name}.pkl")
    return evaluate_results(file_path)

# ----------------------------
# 主程式：CLI 執行
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_name", type=str, required=True, help="模型推理檔名，不含副檔名")
    args = parser.parse_args()

    file_path = os.path.join("outputs", "貸款申請書", f"{args.pred_name}.pkl")
    if not os.path.exists(file_path):
        print(f"找不到檔案: {file_path}")
        sys.exit(1)

    accuracy_results = evaluate_results(file_path)

    if accuracy_results:
        for key, value in accuracy_results.items():
            print(f"{key}: {value:.2%}")
        overall = np.mean(list(accuracy_results.values()))
        print(f"\n平均準確率: {overall:.2%}")
    else:
        print("無法計算準確度")

# python evaluation_貸款申請書.py --pred_name=deepseek_vl2