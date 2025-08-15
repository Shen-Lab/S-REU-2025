This repository contains the code and setup instructions for building an environment to explore cross-domain learning on proteins and biomedical texts, as part of the Multilingual-Molecules: Cross-Domain Learning Across Proteins and Biomedical Texts project. This was adapted from the Pannot LLM (https://github.com/Antoninnnn/Pannot/blob/master/README.md)

This script loads a Pannot-LLaMA model, reads ProteinGym metadata, and for each selected protein CSV builds a 2-shot prompt to classify whether a mutation preserves or improves function (Yes/No). It prints accuracy metrics, and—if you set an output path—saves a CSV with per-row predictions (DMS_id, function, mutation, true label, predicted label, decoded text, mutated sequence).

