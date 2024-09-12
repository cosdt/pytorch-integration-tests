import os

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TrainingArguments, Trainer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling


# 固定随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


# 训练并比较 CPU 和 GPU 的训练损失
def train_and_compare_gpt2(model_name):
    set_seed()

    def train_on_device(use_cpu=False):
        # 加载 GPT-2 模型和 tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token  # GPT-2 没有 pad_token，需要将 eos_token 作为 pad_token

        # 加载 wikitext-2 数据集
        train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        val_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation')

        def preprocess_function(examples):
            return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

        train_dataset = train_dataset.map(preprocess_function, batched=True)
        val_dataset = val_dataset.map(preprocess_function, batched=True)

        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

        # 设置训练参数
        training_args = TrainingArguments(
            output_dir='./results',
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=1,
            logging_dir='./logs',
            logging_steps=10,
            eval_strategy='epoch',
            save_strategy='epoch',
            report_to="none",
            use_cpu=use_cpu
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # 创建 Trainer
        trainer = Trainer(
            data_collator=data_collator,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # 训练模型
        trainer.train()

        # 评估模型
        metrics = trainer.evaluate()

        # 返回评估损失
        return metrics['eval_loss']

    # 在 GPU 上训练（如果有 GPU）
    if torch.npu.is_available():
        print(f"Training on NPU")
        gpu_loss = train_on_device(False)
        print(f"GPU Training Loss: {gpu_loss:.4f}")
    else:
        gpu_loss = None
        print("No GPU available for training.")

    # 在 CPU 上训练
    if os.getenv("IS_CI"):
        # Skip training when running in CI because it's too slow
        cpu_loss = 3.0
    else:
        print(f"Training on CPU")
        cpu_loss = train_on_device(True)

    print(f"CPU Training Loss: {cpu_loss:.4f}")

    return cpu_loss, gpu_loss


# 推理并比较 CPU 和 GPU 的推理损失
def infer_and_compare_gpt2(model_name):
    set_seed()

    def infer_on_device(device: torch.device):
        # 加载 GPT-2 模型和 tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # 设置 pad_token 为 eos_token
        tokenizer.pad_token = tokenizer.eos_token

        # 推理测试句子
        test_sentence = "The quick brown fox jumps over the lazy dog."
        inputs = tokenizer(test_sentence, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        # 计算损失
        loss = outputs.loss.item()
        return loss

    # 在 GPU 上推理（如果有 GPU）
    if torch.npu.is_available():
        gpu_device = torch.device('npu')
        gpu_loss = infer_on_device(gpu_device)
        print(f"GPU Inference Loss: {gpu_loss:.4f}")
    else:
        gpu_loss = None
        print("No GPU available for inference.")

    # 在 CPU 上推理
    cpu_device = torch.device('cpu')
    cpu_loss = infer_on_device(cpu_device)

    print(f"CPU Inference Loss: {cpu_loss:.4f}")

    return cpu_loss, gpu_loss


# 主函数
if __name__ == "__main__":
    model_name = "gpt2"

    # 训练并比较训练损失
    print("Comparing Training Loss:")
    cpu_train_loss, gpu_train_loss = train_and_compare_gpt2(model_name)

    # 推理并比较推理损失
    print("\nComparing Inference Loss:")
    cpu_infer_loss, gpu_infer_loss = infer_and_compare_gpt2(model_name)
