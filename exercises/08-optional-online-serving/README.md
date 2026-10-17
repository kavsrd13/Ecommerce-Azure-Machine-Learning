# 08 - Optional freight online serving

This is optional and is **not** run by repository setup or the other seven
exercises. Managed online endpoints incur charges while deployed. Register a
freight model in your own workspace first, review `deploy/endpoint.yml` and
`deploy/deployment.yml` for model-version and VM-size assumptions, and delete
your own endpoint when finished. Never copy endpoint keys into Git.

After exercise 03 finishes, register its cloud model using the completed job
ID (replace the placeholders with values for your own workspace):

```bash
az ml model create -n olist-freight-cost -v 1 --type mlflow_model \
  --path "azureml://jobs/JOB_ID/outputs/artifacts/paths/outputs/model/" \
  -g YOUR_RESOURCE_GROUP -w YOUR_WORKSPACE --subscription YOUR_SUBSCRIPTION_ID
```

The copied `endpoint.yml` and `deployment.yml` both use `freight-olist` as a
placeholder endpoint name. Change **both** to the same unique name in your
region before creating an endpoint. Review quota and ongoing VM charges; this
exercise is never run by the default bootstrap.
