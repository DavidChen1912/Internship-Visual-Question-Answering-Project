import os
import gc
import pickle
import pandas as pd
from PIL import Image
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

def run_inference(model_name, data_type, prompt_name, output_name):
    # ----------------------------
    # 自動對應路徑
    # ----------------------------
    model_path = os.path.join("model", model_name)
    label_path = os.path.join("label", f"{data_type}.pkl")
    image_dir = os.path.join("data", data_type)
    prompt_path = os.path.join("prompt", f"{prompt_name}.txt")
    output_path = os.path.join("outputs", data_type, f"{output_name}.pkl")

    # ----------------------------
    # 載入模型與處理器
    # ----------------------------
    print(f"🔧 Loading model from: {model_path}")
    processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model = DeepseekVLV2ForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    # ----------------------------
    # 載入 prompt
    # ----------------------------
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # ----------------------------
    # 載入 label.pkl
    # ----------------------------
    with open(label_path, "rb") as f:
        df = pickle.load(f)

    if "mllm_result" not in df.columns:
        df["mllm_result"] = ""

    # ----------------------------
    # 圖片預處理
    # ----------------------------
    def preprocess_image(image_path, max_size=1440):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        return image

    # ----------------------------
    # 推理主迴圈
    # ----------------------------
    for idx, row in df.iterrows():
        image_path = os.path.join(image_dir, row["filename"])
        image = preprocess_image(image_path)

        # 準備 conversation prompt
        conversation = [
            {"role": "<|User|>", "content": f"<|image|>\n{prompt_template}", "images": [image]},
            {"role": "<|Assistant|>", "content": ""},
        ]

        # 編碼成模型輸入
        inputs = processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            system_prompt=""
        ).to(model.device)

        # 模型推理
        inputs_embeds = model.prepare_inputs_embeds(**inputs)
        outputs = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=inputs.attention_mask,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )

        result = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True).strip()
        df.at[idx, "mllm_result"] = result
        print(f"[{idx}] {row['filename']} done.")

        torch.cuda.empty_cache()
        gc.collect()

    # ----------------------------
    # 儲存結果
    # ----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(df, f)

    print(f"\n 推理完成！結果儲存於：{output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="模型名稱，例如 deepseek-vl2-tiny（對應 model/{model}）")
    parser.add_argument("--data_type", type=str, required=True, help="資料類型，例如 損益表、貸款申請書（對應 label/{data_type}.pkl 和 data/{data_type}/）")
    parser.add_argument("--prompt_name", type=str, required=True, help="prompt 檔名（不含副檔名，對應 prompt/{prompt_name}.txt）")
    parser.add_argument("--output_name", type=str, required=True, help="output 的檔名（不含副檔名，將儲存至 outputs/{data_type}/{output_name}.pkl）")
    args = parser.parse_args()

    run_inference(
        model_name=args.model,
        data_type=args.data_type,
        prompt_name=args.prompt_name,
        output_name=args.output_name
    )

# python inference.py --model=deepseek-vl2-tiny --data_type=損益表 --prompt_name=損益表_v3 --output_name=DeepSeek-VL2-test
