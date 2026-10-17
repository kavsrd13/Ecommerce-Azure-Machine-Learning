# E-commerce Azure Machine Learning course

Clone this repository into an **Azure Machine Learning compute instance** after
creating your Azure ML workspace and notebook compute instance manually. This
repository bundles the original Olist data once and arranges the exercises in
teaching order. It is intended for **noncommercial teaching**; see
[dataset attribution](DATA_ATTRIBUTION.md).

## First-time setup on the compute instance

1. Start the notebook compute instance in Azure ML Studio, open **Notebooks >
   Terminal**, and clone this repository:

   ```bash
   git clone https://github.com/kavsrd13/Ecommerce-Azure-Machine-Learning.git
   cd Ecommerce-Azure-Machine-Learning
   ```

2. Sign in to Azure on the compute instance and check the Azure ML CLI v2
   extension. Run `az extension add --name ml --yes` only if the extension
   lookup reports that it is missing:

   ```bash
   az login --use-device-code
   az extension show --name ml
   ```

3. Run the checked-in setup script to unpack the nine Olist CSVs and create a
   Python environment. Set your own subscription, resource group, and workspace
   values. Azure login/permissions and VM-family quota are required for cloud
   jobs. The script must not create a workspace or notebook compute instance.

   ```bash
   python3 scripts/bootstrap.py --subscription YOUR_SUBSCRIPTION_ID \
     --resource-group YOUR_RESOURCE_GROUP --workspace YOUR_WORKSPACE
   ```

4. Select the kernel registered by the script in Studio. Run the exercise
   notebooks **one cell at a time**; job submission cells start billable Azure
   compute. The bootstrap creates only the shared raw-data asset and, when
   explicitly requested, a scale-to-zero CPU cluster for RAI/sweep. Before
   exercises 06 or 07, rerun the same bootstrap command with
   `--skip-install --create-cluster` after checking quota and charges.

Run the offline source checks at any point with
`python3 -m unittest discover -s tests`. This checks the bundled archive,
exercise entrypoints, notebook syntax and absence of saved outputs without
creating cloud resources.

## Exercise order

| Folder | Goal | Cloud prerequisites |
| --- | --- | --- |
| `exercises/01-eda-feature-engineering/` | Inspect joins and build freight features. | None; the notebook stops before model training. |
| `exercises/02-delivery-demand-hotspots/` | Find geographic order-demand clusters and maps. | Notebook kernel only; Folium map tiles need internet. |
| `exercises/03-freight-training/` | Train freight regression as an Azure ML command job. | Raw data asset `olist-ecommerce-raw:1`. |
| `exercises/04-delivery-time-pipeline/` | Run order preparation and delivery-time regression in a pipeline. | Raw data asset `olist-ecommerce-raw:1`. |
| `exercises/05-freight-experimentation/` | Compare AutoML regression and MLflow-tracked models. | Prepared train/validation/test MLTables; compute quota. |
| `exercises/06-late-delivery-responsible-ai/` | Train/register late-delivery classifier and inspect its RAI dashboard. | Four MLTables and a CPU cluster. |
| `exercises/07-late-delivery-sweep/` | Tune hyperparameters on pre-April/April-May data. | Two MLTables and the CPU cluster. |
| `exercises/08-optional-online-serving/` | Optional freight endpoint and sample app. | Registered freight model; persistent endpoint charges. |

Each exercise has its own entrypoint and instructions. Do not run all notebooks
at once or copy the instructor's old Azure job IDs. Students should normally
use separate workspaces; a shared workspace needs unique per-student asset and
model versions to avoid collisions. If a region lacks `Standard_E4ds_v4` quota,
choose a supported CPU size before creating a cluster or submitting jobs.

## Publication boundaries

The compressed original dataset is in `datasets/olist/Ecommerce Dataset.zip`.
Extracted CSVs, generated splits, model outputs, local Python environments,
credentials, billing snapshots, and personal Azure deployment outputs are not
tracked. The upstream MicrosoftLearning repository is a reference, not a
submodule or part of this repository. The lab originally adapted
[Microsoft Learn's MLOps exercises](https://microsoftlearning.github.io/mslearn-mlops/docs/01-experiment-evaluate-models.html).
