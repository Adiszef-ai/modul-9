"""
Wgrywa wytrenowany model do MinIO / S3-compatible storage.
Użycie: python upload_model.py

Wymagane env:
  S3_ENDPOINT_URL  — np. https://minio.twojadomena.pl
  S3_ACCESS_KEY
  S3_SECRET_KEY
  S3_BUCKET        — domyślnie "adult-income-models"
"""
import glob
import os
import sys

import boto3
from botocore.client import Config

MODELS_DIR = "models"


def get_s3_client():
    endpoint  = os.environ.get("S3_ENDPOINT_URL")
    access    = os.environ.get("S3_ACCESS_KEY")
    secret    = os.environ.get("S3_SECRET_KEY")
    if not all([endpoint, access, secret]):
        print("❌ Brak zmiennych: S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY")
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)
        print(f"✅ Bucket '{bucket}' utworzony")


def main():
    pkl_files  = sorted(glob.glob(f"{MODELS_DIR}/model_v*.pkl"))
    json_files = sorted(glob.glob(f"{MODELS_DIR}/metadata_v*.json"))
    if not pkl_files:
        print("❌ Brak pliku modelu w 'models/'. Uruchom najpierw notebook.")
        sys.exit(1)

    pkl_path  = pkl_files[-1]
    meta_path = json_files[-1]
    pkl_name  = os.path.basename(pkl_path)
    meta_name = os.path.basename(meta_path)
    bucket    = os.environ.get("S3_BUCKET", "adult-income-models")

    s3 = get_s3_client()
    ensure_bucket(s3, bucket)

    # Upload wersjonowany
    s3.upload_file(pkl_path,  bucket, f"models/{pkl_name}")
    s3.upload_file(meta_path, bucket, f"models/{meta_name}")
    print(f"✅ Wgrano: models/{pkl_name}")
    print(f"✅ Wgrano: models/{meta_name}")

    # Tag 'latest' — aplikacja pobiera właśnie te klucze
    s3.upload_file(pkl_path,  bucket, "models/model_latest.pkl")
    s3.upload_file(meta_path, bucket, "models/metadata_latest.json")
    print("✅ Tag 'latest' zaktualizowany")
    print(f"\nEndpoint : {os.environ['S3_ENDPOINT_URL']}")
    print(f"Bucket   : {bucket}")


if __name__ == "__main__":
    main()
