# fewshot_with_wildtype.py

import pandas as pd
import torch
import re
from transformers import AutoTokenizer
from pannot.model.builder import load_pretrained_model
from pannot.mm_utils import tokenizer_protein_token, get_model_name_from_path
from pannot.conversation import conv_templates
from pannot.eval.eval_opi.eval_opi_dataset_pannot import KeywordsStoppingCriteria
from sklearn.metrics import confusion_matrix, classification_report

# ===== Model Setup =====
model_path = "/scratch/user/gaygysyz2003/hf_models/pannot/pannot-Meta-Llama-3.1-8B-Instruct-finetune-lora-v00"
model_base = "/scratch/user/gaygysyz2003/hf_models/pannot/pannot-Meta-Llama-3.1-8B-Instruct-pretrain-v00/checkpoint-24000"
model_name = get_model_name_from_path(model_path)

print("Loading model and tokenizer...")
tokenizer, model, _, context_len = load_pretrained_model(
    model_path, model_base, model_name, use_flash_attn=True
)
model.eval()
device = next(model.parameters()).device

# ===== Helper Functions =====
def is_valid_mutation_format(mut):
    pattern = r"^[A-Z]\d+[A-Z]$"
    return all(re.match(pattern, m) for m in mut.split(','))

def get_wildtype_from_row(row):
    seq = list(row["mutated_sequence"])
    for mut in row["mutant"].split(','):
        try:
            wt_aa = mut[0]
            pos = int(mut[1:-1]) - 1
            if 0 <= pos < len(seq):
                seq[pos] = wt_aa
        except:
            continue
    return ''.join(seq)

# ===== DMS Setup =====
dms_ids = ["A0A140D2T1_ZIKV_Sourisseau_2019"] #adjust here and some portions of code to run all 200 csv in ProteinGym
meta_path = "/scratch/user/gaygysyz2003/Pannot/dms_data/DMS_substitutions.csv"
meta_df = pd.read_csv(meta_path)
meta_df = meta_df[meta_df["DMS_id"].isin(dms_ids)].dropna(subset=["DMS_id", "coarse_selection_type"])

# ===== Evaluation Tracking =====
y_true, y_pred, seen_mutated_seqs = [], [], []

# ===== Inference Loop =====
for _, row in meta_df.iterrows():
    dms_id = row["DMS_id"]
    function_name = row["coarse_selection_type"]
    csv_path = f"/scratch/user/gaygysyz2003/Pannot/protein_gym_data/{dms_id}.csv"

    try:
        df = pd.read_csv(csv_path).dropna(subset=["mutated_sequence", "mutant", "DMS_score_bin"])
        df = df[df["DMS_score_bin"].isin([0, 1])]
        df = df[df["mutant"].apply(is_valid_mutation_format)].reset_index(drop=True)

        if len(df) <= 10:
            print(f"Skipping {dms_id}: not enough valid rows.")
            continue

        parts = dms_id.split("_") + ["", "", "", ""]
        gene, species, author, year = parts[:4]
        species_friendly = species.lower().capitalize() if species else "Unknown species"
        friendly_name = f"{gene} protein from {species_friendly}, studied by {author} ({year})"

        conv_mode = "pannot_llama_2"
        conv = conv_templates[conv_mode].copy()
        conv.system = (
            "You are a helpful language and protein assistant. You are able to understand the protein multimodal content that the user provides, and assist the user with a variety of tasks using natural language."
        )
        print("\n====== SYSTEM INSTRUCTION ======")
        print(conv.system)

        few_shot_block = ""
        mutateds = []  # Will collect all 3 mutated sequences (2 few-shot + 1 query)
        for i in range(2):
            row_i = df.loc[i]
            wildtype = get_wildtype_from_row(row_i)
            mutation = row_i["mutant"]
            label = "Yes" if row_i["DMS_score_bin"] == 1 else "No"

            mutated_seq = row_i["mutated_sequence"]
            mutateds.append(mutated_seq)

            example = (
                f"<seq_start><seq><seq_end>\n"
                f"Protein: {friendly_name}\n"
                f"Mutation: {mutation}\n"
                f"Function: {function_name}\n"
                f"Question: Does this mutation preserve or improve function?\n"
                f"Answer: {label}\n\n"
            )
            few_shot_block += example
            print(f"\n--- FEW-SHOT EXAMPLE {i+1} ---")
            print(example.strip())

        row_q = df.loc[10]
        wildtype = get_wildtype_from_row(row_q)
        mutated = row_q["mutated_sequence"]
        mutation = row_q["mutant"]
        true_label = 1 if row_q["DMS_score_bin"] == 1 else 0
        mutateds.append(mutated)

        # ===== Sequence Tower Tokenization for Wildtype + All Mutated Sequences =====
        seqs = []
        seq_attention_masks = []

        # Tokenize wildtype sequence
        print(f"🔬 Wildtype preview: {wildtype[:60]}...")
        wildtype_tokenized = model.get_seq_tower().tokenize(wildtype)
        wildtype_input_id = wildtype_tokenized["input_ids"].to(device)
        wildtype_attn_mask = wildtype_tokenized["attention_mask"].to(device)
        seqs.append(wildtype_input_id)
        seq_attention_masks.append(wildtype_attn_mask)

        for idx, mut_seq in enumerate(mutateds):
            row_label = ['0', '1', '10'][idx] if idx < 3 else str(idx)
            print(f"🔬 Sequence {idx+1} preview (row {row_label}): {mut_seq[:60]}...")
            tokenized = model.get_seq_tower().tokenize(mut_seq)
            seq_input_id = tokenized["input_ids"].to(device)
            seq_attn_mask = tokenized["attention_mask"].to(device)
            seqs.append(seq_input_id)
            seq_attention_masks.append(seq_attn_mask)

        print("✅ Mutated sequence passed to sequence tower:")
        print(mutated[:100], "...")
        print("✅ All sequence input ID shapes:", [s.shape for s in seqs])
        print("✅ All attention mask shapes:", [a.shape for a in seq_attention_masks])

        query_block = (
            f"<seq_start><seq><seq_end>\n"
            f"Protein: {friendly_name}\n"
            f"Mutation: {mutation}\n"
            f"Function: {function_name}\n"
            f"Question: Does this mutation preserve or improve function?\n"
            f"Answer:"
        )

        full_user_message = few_shot_block + query_block
        full_prompt = f"[INST] <<SYS>>\n{conv.system}\n<</SYS>>\n\n{full_user_message} [/INST]"

        print("\n====== FULL PROMPT TO TOKENIZER ======")
        print(full_prompt[:1000] + "..." if len(full_prompt) > 1000 else full_prompt)
        print("==== Prompt Ends With ====")
        print(repr(full_prompt[-200:]))

        input_ids = tokenizer_protein_token(full_prompt, tokenizer, return_tensors="pt").unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        print("Prompt batch shape:", input_ids.shape)

        print("✅ Mutated sequence passed to sequence tower:")
        print(mutated[:100], "...")
        print("✅ All sequence input ID shapes:", [s.shape for s in seqs])
        print("✅ All attention mask shapes:", [a.shape for a in seq_attention_masks])
        print("🧪 EOS token IDs check:")
        print("Yes token ID:", tokenizer.convert_tokens_to_ids(" Yes"))
        print("No token ID:", tokenizer.convert_tokens_to_ids(" No"))
        print("Tokenizer vocab size:", len(tokenizer))
        print("Does ' Yes' exist in vocab?", " Yes" in tokenizer.get_vocab())
        print("Does 'No' exist in vocab?", "No" in tokenizer.get_vocab())
        print("🧪 EOS token IDs check:")
        yes_id = tokenizer.convert_tokens_to_ids(" Yes")
        no_id = tokenizer.convert_tokens_to_ids(" No")
        if yes_id is None:
            yes_id = tokenizer.convert_tokens_to_ids("Yes")
            print("✔️ Fallback to 'Yes' (no space):", yes_id)
        if no_id is None:
            no_id = tokenizer.convert_tokens_to_ids("No")
            print("✔️ Fallback to 'No' (no space):", no_id)
        if None in [yes_id, no_id]:
            raise ValueError(f"❌ Could not find token IDs for Yes/No. Got: Yes={yes_id}, No={no_id}")
        eos_token_ids = [yes_id, no_id]
        print("✅ Final EOS token IDs:", eos_token_ids)

        with torch.no_grad():
            output_ids = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                seqs=seqs,
                seq_attention_mask=seq_attention_masks,
                do_sample=False,
                max_new_tokens=32,
                stopping_criteria=[KeywordsStoppingCriteria([" Yes", " No"], tokenizer, input_ids)],
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                eos_token_id=eos_token_ids
            )
        print("🧪 Output shape:", output_ids.shape)
        print("🧪 Raw decoded tokens:", tokenizer.batch_decode(output_ids, skip_special_tokens=False))
        generated_tokens = output_ids[:, input_ids.shape[-1]:]

        if generated_tokens.shape[-1] == 0:
            print("⚠️ No new tokens from slicing — falling back to full decode.")
            decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
        else:
            decoded = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip().lower()

        print("✅ Decoded prediction:", repr(decoded))
        print("\n====== MODEL PREDICTION ======")
        print(decoded)

        if "yes" in decoded:
            pred = 1
        elif "no" in decoded:
            pred = 0
        else:
            print(f"Skipping {dms_id}: unclear output '{decoded}'")
            continue

        y_true.append(true_label)
        y_pred.append(pred)
        seen_mutated_seqs.append(mutated)

    except Exception as e:
        print(f"Skipping {dms_id} due to error: {e}")
        continue

if len(y_true) > 0:
    print("\n==== Confusion Matrix ====")
    print(confusion_matrix(y_true, y_pred))

    print("\n==== Classification Report ====")
    print(classification_report(y_true, y_pred, labels=[0, 1], target_names=["No", "Yes"], zero_division=0))

    correct = sum([1 if p == t else 0 for p, t in zip(y_pred, y_true)])
    total = len(y_true)
    print("\n==== Accuracy ====")
    print(f"Correct: {correct} / {total}")
    print(f"Accuracy: {(correct / total) * 100:.2f}%")

    print("\n==== Duplicate Sequence Check ====")
    unique = len(set(seen_mutated_seqs))
    print(f"Unique mutated sequences: {unique} / {len(seen_mutated_seqs)}")
    dupes = [seq for seq in seen_mutated_seqs if seen_mutated_seqs.count(seq) > 1]
    print(f"Sample duplicated sequences: {list(set(dupes))[:3]}")
else:
    print("No predictions made — skipping evaluation.")
