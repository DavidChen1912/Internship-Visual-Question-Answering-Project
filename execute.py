import os
import argparse
from inference import run_inference
from evaluation_員工報支 import evaluate_expenses
from evaluation_存摺封面 import evaluate_covers
from evaluation_損益表 import evaluate_income
from evaluation_貸款申請書 import evaluate_loan
from evaluation_資產負債表 import evaluate_balance

def get_next_output_name(output_dir, model_name):
    os.makedirs(output_dir, exist_ok=True)
    base = model_name.replace("/", "_")
    i = 1
    while True:
        filename = f"{base}_test{i}.pkl"
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return filename.replace(".pkl", ""), path
        i += 1

def evaluate_by_data_type(data_type, output_name):
    if data_type == "員工報支":
        return evaluate_expenses(output_name)
    elif data_type == "存摺封面":
        return evaluate_covers(output_name)
    elif data_type == "損益表":
        return evaluate_income(output_name)
    elif data_type == "貸款申請書":
        return evaluate_loan(output_name)
    elif data_type == "資產負債表":
        return evaluate_balance(output_name)
    else:
        raise ValueError(f"不支援的資料類型：{data_type}")

def main(model_name, data_type):
    prompt_name = data_type  # 固定為 prompt/資料類型.txt
    output_dir = os.path.join("outputs", data_type)

    output_name, full_path = get_next_output_name(output_dir, model_name)
    print(f"將輸出結果儲存為：{full_path}")

    # 執行推理
    run_inference(
        model_name=model_name,
        data_type=data_type,
        prompt_name=prompt_name,
        output_name=output_name
    )

    # 評估結果
    print(f"\n開始評估結果：{output_name}")
    acc = evaluate_by_data_type(data_type, output_name)

    for k, v in acc.items():
        print(f"{k}: {v:.2%}")
    print("------")
    print(f"平均準確率: {sum(acc.values()) / len(acc):.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型名稱，例如 deepseek-vl2-tiny")
    parser.add_argument("--data_type", type=str, required=True, choices=["員工報支", "存摺封面", "損益表", "貸款申請書", "資產負債表"])
    args = parser.parse_args()

    main(args.model, args.data_type)

# python execute.py --model=deepseek-vl2-tiny --data_type=損益表
