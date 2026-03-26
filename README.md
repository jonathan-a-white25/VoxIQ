# VoxIQ — Customer Intelligence Platform

A dark-mode Streamlit app that scores customer reviews using a fine-tuned DistilBERT model and presents an interactive insights dashboard.

---

## Features

- **Auto Column Detection** — upload any CSV, VoxIQ identifies text, rating, product, date columns automatically
- **AI Scoring** — fine-tuned DistilBERT (95%+ accuracy) scores every review with sentiment + confidence
- **Customer Insights Dashboard** — KPIs, sentiment breakdown, ratings distribution, top products, flagged negatives
- **Product Performance Dashboard** — avg rating by product, sentiment by product, confidence vs rating scatter, product table
- **Filters** — sidebar filters by sentiment, rating, and product across all dashboard pages
- **Download** — export scored or filtered CSV at any time

---

## Project Structure

```
voxiq/
├── app.py              # Main Streamlit application
├── style.css           # Dark mode stylesheet
├── requirements.txt    # Python dependencies
└── README.md
```

---

## AWS Setup Required

The model loads from S3 at runtime. You need:
- AWS credentials configured (via IAM role if on EC2/SageMaker, or ~/.aws/credentials locally)
- S3 bucket: `operationcapstone-models`
- Model files at: `s3://operationcapstone-models/distilbert_finetuned/`

---

## Local Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Deployment (AWS EC2)

1. Launch an EC2 instance (t3.medium for CPU inference, g4dn.xlarge for GPU)
2. Clone repo and install requirements
3. Attach IAM role with S3 read access
4. Run: `streamlit run app.py --server.port 8501`
5. Open port 8501 in the EC2 security group

---

## Tier 2 Automation (S3 Drop)

To enable automated scoring when a client drops a new CSV into S3:
1. Create a client folder in S3: `s3://operationcapstone-models/clients/<client_id>/incoming/`
2. Deploy the Lambda function (see `lambda/`) triggered by S3 PutObject events
3. Lambda scores the file and saves results to `clients/<client_id>/scored/`
4. Dashboard reads from the scored folder automatically

---

## Model

- Base: `distilbert-base-uncased`
- Fine-tuned on: 200k Amazon Fashion reviews
- Labels: rating ≥ 4 = positive, rating ≤ 2 = negative
- Test accuracy: 95.78% | F1 Macro: 93.10%
