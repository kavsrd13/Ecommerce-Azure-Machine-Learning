# 04 - Order-level delivery-time pipeline

This exercise reuses `olist-ecommerce-raw:1`. Submit the cloud pipeline from
the repository root:

```bash
python scripts/submit.py delivery
```

The pipeline first runs
`src/prep/prep.py`, then trains with `src/train/train.py` on serverless Azure
ML compute. One order, not one order item, is the model's observation.
Actual delivery timestamps must never be checkout-time predictor columns.
