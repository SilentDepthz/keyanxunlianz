import os
import sys
import logging
import datasets
import evaluate

import pandas as pd
import numpy as np

from transformers import AutoModelForSequenceClassification, DebertaV2Tokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import PromptTuningConfig, get_peft_model, TaskType  # 保持不变
from sklearn.model_selection import train_test_split

# --- 数据加载 ---
train = pd.read_csv("/kaggle/input/labeledtraindata-tsv/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("/kaggle/input/testdata-tsv/testData.tsv", header=0, delimiter="\t", quoting=3)

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info(r"running %s" % ''.join(sys.argv))

    # --- 数据集准备 ---
    train, val = train_test_split(train, test_size=.2, random_state=42)

    train_dict = {'label': train["sentiment"], 'text': train['review']}
    val_dict = {'label': val["sentiment"], 'text': val['review']}
    test_dict = {"text": test['review']}

    train_dataset = datasets.Dataset.from_dict(train_dict)
    val_dataset = datasets.Dataset.from_dict(val_dict)
    test_dataset = datasets.Dataset.from_dict(test_dict)

    # --- Tokenizer 和模型 ID ---
    model_id = "microsoft/deberta-v3-base"
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_id)


    # --- 预处理函数 ---
    def preprocess_function(examples):
        return tokenizer(examples['text'], max_length=256, truncation=True)


    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_val = val_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # --- 模型加载 ---
    # 依然需要 num_labels=2 来创建分类头
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    # --- PEFT 配置 (Prompt Tuning) ---
    # 【修改点 1】: 移除无效的 'modules_to_save' 参数
    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_CLS,
        num_virtual_tokens=10
    )

    # 应用 PEFT 配置 (这会默认冻结所有参数)
    model = get_peft_model(model, peft_config)

    # 【修改点 2】: 手动解冻分类头和 Pooler 的参数
    # 这是新的关键步骤，用来解决50%准确率问题
    for name, param in model.named_parameters():
        # 正如警告中提示的，我们需要训练 'classifier' 和 'pooler'
        if 'classifier' in name or 'pooler' in name:
            param.requires_grad = True

    # 打印可训练参数，确认分类头的参数现在是可训练的
    # 你会看到可训练参数量 = 虚拟token的参数 + 分类头/Pooler的参数
    model.print_trainable_parameters()

    # --- 评估指标 ---
    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)


    # --- 训练参数 ---
    # 【修改点 3】: 保持这些重要参数
    training_args = TrainingArguments(
        output_dir='./checkpoint',
        num_train_epochs=4,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,  # 减少日志打印频率
        report_to="none",
        save_strategy="no",
        eval_strategy="epoch",

        # 必须指定学习率，否则默认为0，模型不会学习
        learning_rate=2e-5,
        # 强烈建议开启，防止OOM
        fp16=True,
    )

    # --- 训练器 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --- 开始训练 ---
    logger.info("开始训练...")
    trainer.train()

    # --- 预测测试集 ---
    logger.info("开始预测...")
    prediction_outputs = trainer.predict(tokenized_test)
    test_pred = np.argmax(prediction_outputs[0], axis=-1).flatten()
    print(test_pred)

    # --- 保存结果 ---
    os.makedirs("./result", exist_ok=True)
    result_output = pd.DataFrame(data={"id": test["id"], "sentiment": test_pred})
    result_output.to_csv("./result/deberta_prompt_tuning.csv", index=False, quoting=3)
    logger.info('结果已保存到 ./result/deberta_prompt_tuning.csv')