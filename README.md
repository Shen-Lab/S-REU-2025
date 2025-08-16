# Multimodal Foundation Models for Protein Prediction
This repository contains the code and instructions for using the multimodal MLP and multitask predictor.

combine.py is the code used to concatenate esm2 and esm-if embeddings (npy format) together, creating the "_combined.npy" files that the MLP and multitask predictor use.

combinedMLP.py is the single-output MLP that uses the combined embeddings from ems2 and esm-if. The direcotry should have both the "_combined.npy" and the "labels.npy" files.

multitask_predictor.py is the multitask predictor model, using the combined embeddings and labels (both in a .npy file format), a label csv file (mapped_pathogenicity_threeLabel.csv) and the DMS_substitutions file from ProteinGym (DMS_substitutions.csv).

environment.yml is the environment that was used. you can import it into hprc using: 
```conda env create -n myenv -f environment.yml```

The embeddings for the combined.npy files are in the following zenodo
https://zenodo.org/records/16702612?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImJiZGFjYjBjLTZjMTctNGUwNC04ZGM0LWUyMjRiNDNkODM3OCIsImRhdGEiOnt9LCJyYW5kb20iOiJjNTgwN2M0NmJjNzMyY2I1YmI4MmNjOWY3MDJkNWZmZSJ9.f91sBkOiQQQsmmxSCZUX29kz_WsfwxN-VCfrdXGwmctYkIh8skZ9j6uSz-ofx0GFoQZb7k6NwIuPRBIDtR_PAg

# Directory layout

Run the scripts from the folder that contains them (same directory as combinedMLP.py and multitask_predictor.py). Create these folders exactly as expected by the code:

## For combinedMLP.py

./finalem/                # embeddings (per assay)

    <assay>_combined.npy
./esm2em/                 # labels (per assay)

    <assay>_labels.npy

Each <assay> must exist in both places with matching sample counts.

## For multitask_predictor.py

../../MultimodalStuff/finalem/               # per-assay CSVs + matching embeddings
    <name>.csv
    <name>_combined.npy
../Structure-informed_PLM/mastercsv/         # metadata CSVs
    DMS_substitutions.csv
    mapped_pathogenicity_threeLabels.csv

The script looks for CSV files in ../../MultimodalStuff/finalem/ and loads the same-named _combined.npy next to each CSV.

The two metadata CSVs must be in ../Structure-informed_PLM/mastercsv/ relative to the script.
