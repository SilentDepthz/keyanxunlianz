import os
import sys
import logging
import datasets
import evaluate

import pandas as pd
import numpy as np


from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, \
    Trainer
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split



# --- 1. 数据加载 ---
train = pd.read_csv("/kaggle/input/labeledtraindata-tsv-zip/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/testdata-tsv-zip/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # --- 2. 数据准备 ---
    train, val = train_test_split(train, test_size=.2, random_state=42)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)


    # 这个模型不需要 8-bit 就能在 Kaggle T4 上运行
    model_id = "microsoft/deberta-v3-base"


    tokenizer = AutoTokenizer.from_pretrained(model_id)


    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, max_length=512)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- (修复) 4. 简化模型加载 (不再需要 8-bit) ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=2,
        # (修复) 我们不需要 device_map="auto"，因为模型很小
    )
    # (修复) 删除了所有 BitsAndBytesConfig 和 quantization_config

    # --- 5. 添加 LoRA (这仍然是高效的) ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    # (修复) 删除了 prepare_model_for_kbit_training
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 6. 训练设置 ---
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=3,
        per_device_train_batch_size=8,  # 我们可以用更大的 Batch Size
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        bf16=True,  # 开启混合精度

        learning_rate=5e-6,

        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        report_to="none",
        save_strategy="no",
        eval_strategy="epoch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    # --- 7. 保存结果 ---
    output_dir_path = "./result"
    os.makedirs(output_dir_path, exist_ok=True)

    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv(os.path.join(output_dir_path, "deberta_v3_base_lora.csv"), index=False, quoting=3)
    logging.info('result saved!')