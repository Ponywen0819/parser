from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "HuggingFaceTB/SmolLM3-3B"

device = "cuda"  # for GPU usage or "cpu" for CPU usage
# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.float16,trust_remote_code=True).to(device)
# # prepare the model input
prompt = "13651+75615"
messages_think = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages_think,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
# Generate the output
import time
start = time.time()
print("Generating response...")
generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
# Get and decode the output
output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :]
print(tokenizer.decode(output_ids, skip_special_tokens=True))
print(f"Time taken: {time.time() - start:.2f} seconds")
print("token per second:", len(output_ids) / (time.time() - start))