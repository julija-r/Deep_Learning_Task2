from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

# quantization config same as in traning
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# base model and fine tuned
base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter_path = "models/lithuanian-slang"

# same system prompt used during training
system_prompt = (
    "Tu esi lietuviškai kalbantis asistentas, skirtas suprasti nusikaltėlių žargoną. "
    "Tavo užduotis – paaiškinti sakinio prasmę, atsižvelgiant į žargoną. "
)

# example input
sakinys = "Šiandien užvežė sniego."
zodis = "sniego"

#request in the same format as training
request = f"Paaiškink žodį {zodis} tekste: {sakinys}"

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": request},
]

# same function
def generate_answer(model, tokenizer, messages):
    # convert chat messages to model input string
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # tells model to generate assistant reply
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
        )

    # decode tokens back to readable text
    return tokenizer.decode(output[0], skip_special_tokens=True)


# load tokenizer (same as base model)
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("loading base model")

# load base model
base = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
)
base.eval()

# test base model
print("BEFORE FINE-TUNING")
print(generate_answer(base, tokenizer, messages))


# load lora adapter on top
finetuned = PeftModel.from_pretrained(base, adapter_path)
finetuned.eval()

# test fine-tuned model
print("AFTER FINE-TUNING")
print(generate_answer(finetuned, tokenizer, messages))