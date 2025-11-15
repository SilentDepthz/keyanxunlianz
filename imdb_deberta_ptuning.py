import os
import sys
import logging
import datasets
import evaluate

import pandas as pd
import numpy as np
import torch  # 确保导入了 torch

# 移除了重复的 AutoModelForSequenceClassification 导入
from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import BitsAndBytesConfig, Trainer, TrainingArguments
from peft import PromptEncoderConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split

# --- 1. 加载数据 (这部分不变) ---
train = pd.read_csv("/kaggle/input/labeledtraindata-tsv/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/testdata-tsv/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # --- 2. 数据集准备 (这部分不变) ---
    train, val = train_test_split(train, test_size=.2)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    # --- 3. Tokenizer 和数据处理 (这部分不变) ---
    model_id = "microsoft/deberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


    def preprocess_function(examples):
        return tokenizer(examples['text'], max_length=256, truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 4. 模型加载和 PEFT (## --- 这里是主要修复 ---) ---

    # B. 加载量化后的基础模型 (只加载一次)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"  # 使用8-bit时，强烈推荐使用 "auto"
    )

    # C. 定义 P-Tuning 配置
    peft_config = PromptEncoderConfig(
        num_virtual_tokens=20,
        encoder_hidden_size=128,
        task_type=TaskType.SEQ_CLS
    )

    # D. (## --- 修复: 应用 PEFT 配置 ---)
    #    我们用 peft_config 包装 'model'，并把结果存回 'model' 变量中
    model = get_peft_model(model, peft_config)

    # E. (## --- 修复: 现在在 PEFT 模型上调用 ---)
    #    这将显示只有一小部分参数是可训练的 (P-Tuning 虚拟 token)
    model.print_trainable_parameters()

    # --- 5. 训练器设置 (Metrics 和 TrainingArguments 不变) ---
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        per_device_eval_batch_size=4,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        fp16=True,
        logging_dir='./logs',  # directory for storing logs
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        eval_strategy="epoch"
    )

    # --- 6. 训练器初始化 (## --- 关键 ---) ---
    #    现在 'model' 变量是 PEFT 包装过的模型
    #    Trainer 将只训练 P-Tuning 的参数
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=tokenized_train,  # training dataset
        eval_dataset=tokenized_val,  # evaluation dataset
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 7. 训练和预测 (这部分不变) ---
    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    # --- 8. 保存结果 (## --- 修复: 添加了文件夹创建和重命名 ---) ---

    # (## --- 修复: 确保 ./result 文件夹存在 ---)
    os.makedirs("./result", exist_ok=True)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})

    # (## --- 修复: 重命名文件以匹配 "ptuning" ---)
    output_csv_path = "./result/deberta_ptuning_int8.csv"
    result_output.to_csv(output_csv_path, index=False, quoting=3)

    logging.info(f'result saved to {output_csv_path}!')