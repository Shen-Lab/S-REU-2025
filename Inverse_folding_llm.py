import csv
import re
from datetime import datetime
from transformers.generation.stopping_criteria import StoppingCriteria
from pannot.model.builder import load_pretrained_model
from pannot.mm_utils import tokenizer_protein_token, get_model_name_from_path
from pannot.conversation import conv_templates
import torch
from pathlib import Path
import pandas as pd

# === CONFIG ===
CSV_INPUT = "/scratch/user/jawadnelhassan2005/LLM/Pannot/csv_output/Structural_sequences.csv" # <------ will be different for whoever uses the code
#  you will need to have Structural_sequences.csv (this is a csv with every csv of casp 15's 3di sequences and amino acid sequence)
VERSION = "E"
CSV_OUTPUT = "/scratch/user/jawadnelhassan2005/LLM/Pannot/csv_output/predicted_outputs.csv" # <------ will be different for who ever uses the code, you will have to make an output directory named csv_output
VERSION = "E"
NUM_ATTEMPTS = 100  # The number of times this will repeat the code to get an output

# === Load Model ===
model_path = "/scratch/user/jawadnelhassan2005/LLM/Pannot/checkpoints/llama3_lora_v02/pannot-Meta-Llama-3.1-8B-Instruct-finetune-lora-v02" #<------ will be different for who ever uses the code
VERSION = "E"
model_base = "/scratch/user/jawadnelhassan2005/LLM/Pannot/checkpoints/llama3_lora_v02/pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v02/checkpoint-25053"  #<------ will be different for who ever uses the code
model_name = get_model_name_from_path(model_path)
tokenizer, model, *_ = load_pretrained_model(model_path, model_base, model_name, use_flash_attn=True)
model.eval()
device = next(model.parameters()).device

# === Prompt Setup ===
SEQ_START, SEQ_END = "[", "]"
wrap = lambda s: f"{SEQ_START}{s}{SEQ_END}"
AA_CHARS = "ACDEFGHIKLMNPQRSTVWY"
def build_rules(length: int) -> str:
    return (
        "The above is a 3Di sequence in lowercase.\n"
        "Each lowercase character represents a discrete backbone geometry state (20 total).\n"
        f"Your task is to reconstruct the amino acid sequence of exactly {length} uppercase characters.\n"
        f"Output must start with {SEQ_START} and end with {SEQ_END}, be exactly {length} characters long.\n"
        f"Use ONLY one-letter amino acid codes ({', '.join(AA_CHARS)}). Stop after the closing bracket.\n"
    )

# === Read Training Data ===
# === Read CSV and Select Query + Training ===
df = pd.read_csv(CSV_INPUT)

# First sample (row 0) is the query  <--- this is what you will be the sequence you are trying to predict
query = {"input": df.iloc[0]["3di"].lower()}
queryLength = len(query["input"])

# Next six samples (rows 1 to 6) are training <--- this is what you will be the sequences you are giving examples for training, so far you can probably use up to 7 tokens that won't 
query = {"input": df.iloc[0]["3di"].lower()}
train_data = [
    {"input": row["3di"].lower(), "label": wrap(row["amino_acid"])}
    for _, row in df.iloc[1:7].iterrows()
]

# === Stop Criterion ===
end_ids = tokenizer.encode(" ]", add_special_tokens=False)
stop_token_ids = torch.tensor(end_ids, device=device)
class StopOnClose(torch.nn.Module):
    def __call__(self, input_ids, scores, **kwargs):
        return (input_ids[0, -len(stop_token_ids):] == stop_token_ids).all()
stopper = [StopOnClose()]

# === Header ===
with open(CSV_OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Attempt", "Version", "PredictedSeq", "ExpectedLen", "PredictedLen", "Match", "Timestamp"])

# === Prompt Loop ===
conv_mode = ("pannot_llama_2" if "llama" in model_name.lower() else "pannot_v0")
conv_template = conv_templates[conv_mode]
lengths = []

for attempt in range(1, NUM_ATTEMPTS + 1):
    conv = conv_template.copy()
    for ex in train_data:
        conv.append_message(conv.roles[0], f"{ex['input']}\n\n{build_rules(len(ex['input']))}")
        conv.append_message(conv.roles[1], ex["label"])
    conv.append_message(conv.roles[0], f"{query['input']}\n\n{build_rules(queryLength)}")
    conv.append_message(conv.roles[1], None)

    # Truncate if too long
    while True:
        prompt = conv.get_prompt()
        if len(tokenizer(prompt).input_ids) <= 2048:
            break
        conv.messages.pop(0)
        conv.messages.pop(0)

    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            inputs=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=queryLength + 2,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            num_beams=1,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            stopping_criteria=stopper,
        )

    raw = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)
    matches = re.findall(r"\[([ACDEFGHIKLMNPQRSTVWY]{10,})]", raw)
    bestSeq = max(matches, key=len) if matches else ""
    predictedLen = len(bestSeq)
    lengths.append(predictedLen)

    with open(CSV_OUTPUT, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            attempt,
            VERSION,
            bestSeq,
            queryLength,
            predictedLen,
            predictedLen == queryLength,
            datetime.now().isoformat()
        ])

# === Final Report ===
avgLength = sum(lengths) / NUM_ATTEMPTS
print(f"\n✅ Finished {NUM_ATTEMPTS} runs. Average Length: {avgLength:.2f}")

