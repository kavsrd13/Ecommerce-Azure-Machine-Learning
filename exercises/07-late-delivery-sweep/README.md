# 07 - Azure ML sweep job (300-minute limit)

Open `Azure ML late delivery sweep.ipynb` in the Azure ML notebook compute
instance with the root **Python (Olist Azure ML)** kernel, after running root
`scripts/bootstrap.py` to extract shared CSVs and configure the workspace.
`prepare.py` defaults to that shared CSV folder and supports `--raw-data`.
The notebook prepares its own April-May validation asset and tunes logistic
regression with at most 8 cloud trials, 1 concurrent trial, a 30-minute trial
timeout and a 300-minute overall sweep timeout. It does not use June-August
test data for selecting hyperparameters. A completed sweep is not a deployment
approval; inspect late-class precision/recall as well as average precision.
Registration and sweep submission require explicit opt-in; the returned job ID
is saved in ignored `.sweep-run-id.json` to prevent duplicate submissions.
Set `REGISTER_RESOURCES = True` in the notebook for a new workspace, then
review its limits before deliberately setting `SUBMIT_SWEEP = True`.
