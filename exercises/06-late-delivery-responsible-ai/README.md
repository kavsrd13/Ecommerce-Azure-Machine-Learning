# 06 - Late-delivery classification and Responsible AI

Follow `lab-guide.html` on a Linux Azure ML compute instance. Root
`scripts/bootstrap.py` extracts shared Olist CSVs, writes ignored workspace
`config.json`, and optionally creates the scale-to-zero `olist-rai-cpu` cluster
with `--create-cluster`. `prepare.py` defaults to those shared CSVs (or use
`--raw-data` for a different directory). Register
the four Parquet-backed MLTable assets, submit `jobs/train.yml`, register its
sklearn-flavored MLflow model, and submit `jobs/rai-dashboard.yml`. The
cluster is never created automatically by the exercise. Never claim geographic cohort
error differences establish causality or fairness violations.
