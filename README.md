# Internship – E.SUN Commercial Bank  
<img src="docs/Esun-logo.png" alt="E.SUN Bank Logo" width="80%"/>
**Visual Question Answering Project**

## ⚠️ Disclaimer  
This repository documents my internship project at **E.SUN Commercial Bank**.  
Since the project involves **image-based proprietary datasets**, no open-source substitutes are available.  
This repository only provides **illustrative examples and code structure**, without including any company data.  

---

## 📖 Introduction  
With the rapid evolution of **vision-language models (VLLMs)**, organizations often face uncertainty about how different models perform on **specific tasks**.  
At E.SUN Bank, existing evaluation methods were unsuitable for internal datasets and business scenarios, and there was no automated pipeline to streamline the process.  

To address this, our team built an **end-to-end evaluation pipeline** that integrates:  
- image input handling  
- prompt engineering  
- LLM response generation  
- task-specific scoring  
- automatic leaderboard updating  

This system enables internal users to quickly benchmark multiple models with **one command**.  
The full project pipeline and documentation can be found under the [`/docs`](./docs) directory.  

---

## 🛠️ Scripts  

- **`execute.py`**: Runs the full evaluation workflow (end-to-end).  
- **`inference.py`**: Performs model inference on test data and saves raw outputs.  
- **`evaluation_員工報支.py`**: Evaluation for invoice tasks.  
- **`evaluation_存摺封面.py`**: Evaluation for bankbook cover tasks.  
- **`evaluation_損益表.py`**: Evaluation for income statement tasks.  
- **`evaluation_貸款申請書.py`**: Evaluation for loan application tasks.  
- **`evaluation_資產負債表.py`**: Evaluation for balance sheet tasks.  

---

## 🚀 Usage  

Run the full pipeline with a single command:  

```
python execute.py --model=deepseek-vl2-tiny --data_type=損益表
```

#### Notes:
- **`model`** specifies the model to be tested.
- **`data_type`** specifies the task (e.g., invoice, passbook, income statement).
- The one-click execution uses generalized parameters. For more customized testing, run inference and evaluation separately as shown below.

## 🔎 Inference

Run inference only (no evaluation):

```
python inference.py --model=deepseek-vl2-tiny --data_type=損益表 --prompt_name=損益表_v3 --output_name=DeepSeek-VL2-test
```

#### Notes:
- Each task may support multiple prompts; choose the one that fits your experiment.
- **`output_name`** determines the filename saved in [`/outputs`](./outputs).

## 📊 Evaluation

Run evaluation on saved outputs in [`/outputs`](./outputs):

```
# Evaluate invoice task
python evaluation_員工報支.py --pred_name=deepseek_vl2

# Evaluate passbook cover
python evaluation_存摺封面.py --pred_name=deepseek_vl2

# Evaluate income statement
python evaluation_損益表.py --pred_name=deepseek_vl2

# Evaluate loan application
python evaluation_貸款申請書.py --pred_name=deepseek_vl2

# Evaluate balance sheet
python evaluation_資產負債表.py --pred_name=deepseek_vl2
```

#### Notes:
- **`pred_name`** should match the output filename in [`/outputs`](./outputs).





