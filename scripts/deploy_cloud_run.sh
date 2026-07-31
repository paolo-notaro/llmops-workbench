#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 PROJECT_ID RUNTIME_SERVICE_ACCOUNT [REGION]" >&2
  exit 2
fi

project_id="$1"
runtime_service_account="$2"
region="${3:-europe-west1}"

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  --project "$project_id"

if ! gcloud iam service-accounts describe "$runtime_service_account" --project "$project_id" >/dev/null; then
  echo "Runtime service account '$runtime_service_account' must exist in project '$project_id'." >&2
  echo "The deploying principal must also have Service Account User permission on it." >&2
  exit 1
fi

gcloud run deploy llmops-workbench \
  --quiet \
  --source . \
  --project "$project_id" \
  --region "$region" \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 1 \
  --memory 1Gi \
  --concurrency 20 \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 60 \
  --cpu-throttling \
  --service-account "$runtime_service_account" \
  --set-env-vars LLM_PROVIDER=mock,LOG_LEVEL=INFO \
  --labels app=llmops-workbench,environment=demo \
  --description "Public LLMOps evaluation and RAG observability workbench"
