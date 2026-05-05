from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch

# load dataset from csv file
train_dataset = load_dataset("csv", data_files="data/zodynas_su_sakiniais.csv", split="train")

# base model to finetune
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
HF_TOKEN = None

# load tokenizer and set padding
tokenizer = AutoTokenizer.from_pretrained(
    base_model,
    trust_remote_code=True,
    token=HF_TOKEN,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# system instruction that will be included in every sample
system_prompt = (
    "Tu esi lietuviškai kalbantis asistentas, skirtas suprasti nusikaltėlių žargoną. "
    "Tavo užduotis – paaiškinti sakinio prasmę, atsižvelgiant į žargoną. "
)

# converting dataset rows into chat format expected by the model
def format_chat_template(batch):
    samples = []
    for sakinys, zodis, paaiskinimas in zip(batch["sakinys"], batch["zodis"], batch["paaiskinimas_new"]):
        # build user request from sentence + word
        request = f"Paaiskink zodi {zodis} tekste: {sakinys}"

        # structure conversation
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
            {"role": "assistant", "content": paaiskinimas},
        ]

        # convert to plain text using tokenizers chat template
        text = tokenizer.apply_chat_template(
            row_json,
            tokenize=False,
            add_generation_prompt=False,
        )
        samples.append(text)

    return {"text": samples}

# applying formatting to entire dataset
train_dataset = train_dataset.map(format_chat_template, batched=True)

# 4bit quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# load base model with quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)

# prepare model
model = prepare_model_for_kbit_training(model)

# LORA configs
lora_config = LoraConfig(
    r=256,
    lora_alpha=512,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


# training configs
sft_config = SFTConfig(
    output_dir="models/lithuanian-slang",
    num_train_epochs=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    dataset_text_field="text")

# initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=lora_config,
    args=sft_config,
)

# start training
trainer.train()

# save trained adapters nd tokenizer
trainer.save_model("models/lithuanian-slang")
tokenizer.save_pretrained("models/lithuanian-slang")

print("Done")