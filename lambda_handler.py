"""
VoxIQ — Lambda Handler: trigger weekly SageMaker retraining job

Trigger: EventBridge (CloudWatch Events) rule
  Schedule expression: cron(0 2 ? * MON *)   # Every Monday at 02:00 UTC

Required IAM permissions for the Lambda execution role:
  - sagemaker:CreateTrainingJob
  - s3:GetObject, s3:PutObject, s3:ListBucket  on s3://operationcapstone-models/*
  - iam:PassRole  for SAGEMAKER_ROLE_ARN

Required environment variables:
  SAGEMAKER_ROLE_ARN   — IAM role SageMaker assumes during the training job
  SOURCE_S3_URI        — S3 URI of the source tarball containing retrain.py
                         e.g. s3://operationcapstone-models/source/retrain.tar.gz
"""

import json
import os
from datetime import datetime, timezone

import boto3

BUCKET           = "operationcapstone-models"
REGION           = os.environ.get("AWS_REGION", "us-east-1")
SAGEMAKER_ROLE   = os.environ["SAGEMAKER_ROLE_ARN"]
SOURCE_S3_URI    = os.environ["SOURCE_S3_URI"]
OUTPUT_S3_URI    = f"s3://{BUCKET}/sagemaker_output/"
INSTANCE_TYPE    = "ml.g4dn.xlarge"
MAX_RUNTIME_SECS = 7200  # 2 hours


def handler(event, context):
    sm = boto3.client("sagemaker", region_name=REGION)

    timestamp  = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    job_name   = f"voxiq-retrain-{timestamp}"

    request = {
        "TrainingJobName": job_name,
        "RoleArn": SAGEMAKER_ROLE,
        "AlgorithmSpecification": {
            # PyTorch 2.1 managed container (GPU, Python 3.10)
            "TrainingImage": (
                f"763104351884.dkr.ecr.{REGION}.amazonaws.com/"
                "pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
            ),
            "TrainingInputMode": "File",
        },
        "HyperParameters": {
            "sagemaker_program":        "retrain.py",
            "sagemaker_submit_directory": SOURCE_S3_URI,
        },
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType":             "S3Prefix",
                        "S3Uri":                  f"s3://{BUCKET}/training_data/",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "ContentType": "text/csv",
            }
        ],
        "OutputDataConfig": {
            "S3OutputPath": OUTPUT_S3_URI,
        },
        "ResourceConfig": {
            "InstanceType":   INSTANCE_TYPE,
            "InstanceCount":  1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": MAX_RUNTIME_SECS,
        },
        "EnableManagedSpotTraining": False,
    }

    response = sm.create_training_job(**request)
    arn      = response["TrainingJobArn"]

    print(f"Started SageMaker training job: {job_name}")
    print(f"ARN: {arn}")

    return {
        "statusCode": 200,
        "body": json.dumps({
            "job_name": job_name,
            "arn":      arn,
        }),
    }
