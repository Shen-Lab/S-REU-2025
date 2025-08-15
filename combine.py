import os
import numpy as np
from tqdm import tqdm

def get_wildtype_id(mutant_filename):
    return "_".join(mutant_filename.split("_")[:2])  # e.g. A0A140D2T1_ZIKV

def combine_embeddings(mutant_dir, wildtype_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mutant_files = sorted(f for f in os.listdir(mutant_dir) if f.endswith("_embeddings.npy"))

    for mutant_file in tqdm(mutant_files, desc="Combining embeddings"):
        try:
            # Derive wildtype base name
            wildtype_id = get_wildtype_id(mutant_file)
            wildtype_path = os.path.join(wildtype_dir, wildtype_id + ".npy")
            mutant_path = os.path.join(mutant_dir, mutant_file)

            if not os.path.exists(wildtype_path):
                print(f"[!] Missing wildtype for {mutant_file}")
                continue

            # Load mutant and wildtype
            mutant_embed = np.load(mutant_path)  # (N, 1280)
            wildtype_embed = np.load(wildtype_path)  # (1, L, 512)
            wildtype_mean = wildtype_embed.squeeze(0).mean(axis=0)  # (512,)

            # Repeat for all mutants (N, 512)
            wt_repeated = np.repeat(wildtype_mean[np.newaxis, :], mutant_embed.shape[0], axis=0)

            # Concatenate: (N, 1280 + 512)
            combined = np.concatenate([mutant_embed, wt_repeated], axis=1)

            # Save
            out_name = mutant_file.replace("_embeddings.npy", "_combined.npy")
            out_path = os.path.join(output_dir, out_name)
            np.save(out_path, combined)
            print(f"[✓] Saved: {out_path}")

        except Exception as e:
            print(f"[!] Failed on {mutant_file}: {e}")

if __name__ == "__main__":
    combine_embeddings(
        mutant_dir="esm2em",
        wildtype_dir="structureem",
        output_dir="finalem"
    )