"""Cheap, offline checks students can run immediately after cloning."""

import json
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_CSVS = {
    "olist_customers_dataset.csv", "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv", "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
    "olist_products_dataset.csv", "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
}


class StudentCloneChecks(unittest.TestCase):
    def test_shared_archive_has_only_original_olist_data(self):
        path = ROOT / "datasets/olist/Ecommerce Dataset.zip"
        self.assertLess(path.stat().st_size, 100 * 1024 * 1024)
        with zipfile.ZipFile(path) as archive:
            actual = {name for name in archive.namelist() if not name.endswith("/")}
        self.assertEqual(actual, {f"Ecommerce Dataset/{name}" for name in ORIGINAL_CSVS})

    def test_each_exercise_has_an_entrypoint(self):
        for number in range(1, 9):
            folder = next((ROOT / "exercises").glob(f"{number:02d}-*"))
            with self.subTest(exercise=number):
                self.assertTrue((folder / "README.md").is_file())
        for file in [
            "scripts/bootstrap.py", "scripts/submit.py",
            "exercises/03-freight-training/jobs/train-job.yml",
            "exercises/04-delivery-time-pipeline/pipelines/delivery-time-pipeline.yml",
            "exercises/05-freight-experimentation/prepare.py",
            "exercises/06-late-delivery-responsible-ai/prepare.py",
            "exercises/07-late-delivery-sweep/prepare.py",
        ]:
            with self.subTest(file=file):
                self.assertTrue((ROOT / file).is_file())

    def test_public_notebooks_are_valid_and_have_no_saved_outputs(self):
        for path in (ROOT / "exercises").rglob("*.ipynb"):
            with self.subTest(notebook=str(path.relative_to(ROOT))):
                notebook = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "olist-azure-ml")
                code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
                self.assertTrue(code)
                for cell in code:
                    self.assertIsNone(cell["execution_count"])
                    self.assertEqual(cell["outputs"], [])
                    source = "".join(cell["source"])
                    if not any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
                        compile(source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
