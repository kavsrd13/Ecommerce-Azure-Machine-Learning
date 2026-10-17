# 05 - AutoML and MLflow freight experiments

Run root `scripts/bootstrap.py` for the shared Olist data and workspace
`config.json`, then open `Olist freight Azure ML experiments.ipynb` on the
Linux compute instance with the **Python (Olist Azure ML)** kernel.
The notebook builds chronological MLTable assets, submits bounded Azure AutoML
regression and MLflow-tracked Ridge/boosting command jobs, and evaluates one
selected model. Uploads and billable submissions are disabled until explicitly
enabled; job IDs persist in ignored `.freight-run-ids.json` so rerunning can
reconnect instead of starting duplicates. This lab uses shared Olist CSVs, not
the diabetes data from the MicrosoftLearning reference repository.
For a new student workspace, set `UPLOAD_DATA = True` in notebook Step 1,
run the data-asset cell, then set `SUBMIT_JOBS = True` and run the AutoML,
Ridge and boosting submission cells once. The ignored run-ID file prevents
accidental duplicate submissions on later reruns.
