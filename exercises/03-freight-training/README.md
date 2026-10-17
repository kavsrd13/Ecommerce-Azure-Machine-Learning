# 03 - Freight regression on Azure ML

After `scripts/bootstrap.py` registers `olist-ecommerce-raw:1` in your own
workspace, submit the cloud job from the repository root:

```bash
python scripts/submit.py freight
```

The YAML sends
`src/train.py` and the raw-data asset to a serverless CPU training job.
Evaluate MAE in reais against the median baseline. Do not use the held-out
future months to choose model settings. Cloud compute incurs charges.
