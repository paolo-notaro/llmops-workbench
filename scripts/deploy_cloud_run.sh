#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 PROJECT_ID [REGION]" >&2
  exit 2
fi

project_id="$1"
region="${2:-europe-west1}"

gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  --project "$project_id"

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
  --set-env-vars LLM_PROVIDER=mock,LOG_LEVEL=INFO \
  --labels app=llmops-workbench,environment=demo \
  --description "Public LLMOps evaluation and RAG observability workbench"
