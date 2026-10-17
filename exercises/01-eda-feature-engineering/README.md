# 01 - Explore and engineer Olist features

Open `olist-eda-feature-engineering.ipynb` using the repository's Olist kernel.
Run cells in order. The nine CSVs come from `datasets/olist/Ecommerce Dataset/`
after root `scripts/bootstrap.py` extracts the archive. The goal is to justify
the item-level freight target, join cardinalities, chronology and leakage
checks before a cloud regression model is trained.
The final cell exports processed features under this exercise's ignored
`outputs/` directory, never into the shared raw-data folder.
