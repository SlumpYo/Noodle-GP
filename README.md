## Noodle-GP

This repository implements the Noodle-GP training pipeline for guitar transcription. It uses the **GuitarSet** dataset and the **Tab-estimator** conversion kit to prepare tab+audio pairs, then runs a 6-fold cross-validation training procedure.

## Usage
To use the model, first set up a conda environment using the env.yml file:
```
conda env create -f env.yml
```

Activate the environment:
```
conda activate noodleGP
```

To download and convert GuitarSet see [Tab-Estimator](https://github.com/KimSehun725/Tab-estimator).

To start training, run:
```
python -m src.training.6-fold-final-train
```

This will train the 6 folds.
