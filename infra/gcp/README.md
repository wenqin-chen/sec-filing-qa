# GCP Cloud Run deployment

One container, one region, scale-to-zero. The image is the `full` target built by
`.github/workflows/build.yml` and published to GHCR; Cloud Run pulls only from Artifact Registry,
so the deploy workflow mirrors the GHCR tag into the project's repository first.

Nothing in this directory contains a secret. Vendor keys live in Secret Manager and are mounted
as environment variables with `--set-secrets`.

## One-time project setup (by hand, Day 1)

```bash
export PROJECT_ID=<your-project-id> REGION=us-central1
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
    secretmanager.googleapis.com iamcredentials.googleapis.com

# Artifact Registry repository the deploy workflow pushes into
gcloud artifacts repositories create sec-filing-qa --repository-format=docker \
    --location="$REGION" --description="sec-filing-qa images"

# Secrets (values are entered interactively; never paste them into a shell history)
printf '%s' "$OPENAI_API_KEY"    | gcloud secrets create openai-api-key    --data-file=-
printf '%s' "$ANTHROPIC_API_KEY" | gcloud secrets create anthropic-api-key --data-file=-
printf '%s' "$SECQA_API_KEY"     | gcloud secrets create secqa-api-key     --data-file=-

# Deployer service account + Workload Identity Federation for GitHub Actions (no JSON keys)
gcloud iam service-accounts create secqa-deployer --display-name="sec-filing-qa deployer"
SA="secqa-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
for role in roles/run.admin roles/artifactregistry.writer roles/iam.serviceAccountUser \
            roles/secretmanager.secretAccessor; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:$SA" --role="$role"
done
gcloud iam workload-identity-pools create github --location=global --display-name="GitHub"
gcloud iam workload-identity-pools providers create-oidc github-oidc \
    --location=global --workload-identity-pool=github \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == '<owner>/sec-filing-qa'"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
gcloud iam service-accounts add-iam-policy-binding "$SA" --role=roles/iam.workloadIdentityUser \
    --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/<owner>/sec-filing-qa"
```

Then in the GitHub repository:

| Kind     | Name                     | Value                                                                                   |
|----------|--------------------------|-----------------------------------------------------------------------------------------|
| variable | `DEPLOY_GCP`             | `true`                                                                                  |
| variable | `GCP_PROJECT_ID`         | the project id                                                                          |
| variable | `GCP_REGION`             | `us-central1` (default)                                                                 |
| variable | `GCP_SERVICE_ACCOUNT`    | `secqa-deployer@<project>.iam.gserviceaccount.com`                                      |
| variable | `CLOUD_RUN_SET_SECRETS`  | `OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,SECQA_API_KEY=secqa-api-key:latest` |
| variable | `SECQA_INDEX_URL`        | `https://github.com/<owner>/sec-filing-qa/releases/download/v0.1.0/index-v0.1.tar.zst` (or `gs://...`) |
| variable | `SECQA_EMBEDDER`         | `local` for the published bge-small index, `hashing` for the fixture index              |
| secret   | `GCP_WIF_PROVIDER`       | `projects/<number>/locations/global/workloadIdentityPools/github/providers/github-oidc` |

## First deploy from the laptop (Day 5)

`cloudrun.env.example` lists the service environment; copy it, fill in values, and run:

```bash
IMAGE=ghcr.io/<owner>/sec-filing-qa:<sha>
TARGET="${REGION}-docker.pkg.dev/${PROJECT_ID}/sec-filing-qa/sec-filing-qa:<sha>"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"
docker pull "$IMAGE" && docker tag "$IMAGE" "$TARGET" && docker push "$TARGET"   # or let CI mirror it

gcloud run deploy sec-filing-qa --image "$TARGET" --region "$REGION" \
    --memory 2Gi --cpu 1 --cpu-boost --min-instances 0 --max-instances 2 \
    --concurrency 4 --timeout 120 --port 8080 --allow-unauthenticated \
    --env-vars-file cloudrun.env.yaml \
    --set-secrets OPENAI_API_KEY=openai-api-key:latest,ANTHROPIC_API_KEY=anthropic-api-key:latest,SECQA_API_KEY=secqa-api-key:latest
URL=$(gcloud run services describe sec-filing-qa --region "$REGION" --format 'value(status.url)')
curl -fsS "$URL/readyz"
curl -fsS -X POST "$URL/v1/ask" -H 'Content-Type: application/json' \
    -d '{"question": "What were total net sales in fiscal 2023?", "provider": "mock"}'
```

After that, `.github/workflows/deploy-cloudrun.yml` performs the same steps on every `v*` tag or
by manual dispatch and pastes the smoke output into the job summary.

## Sizing and cost notes

- The `full` image is ~1.6 GB (torch CPU + bge-small); cold start is 10-20 s. `--cpu-boost`
  helps; `--min-instances 1` removes cold starts at roughly the price of one always-on vCPU.
- `--concurrency 4` because one process holds one DuckDB handle and the agent loop is CPU-bound
  for seconds at a time; scale with instances (`--max-instances 2` caps the bill).
- The daily budget (`SECQA_DAILY_BUDGET_USD`) is per instance and in-memory (documented
  limitation); the public demo runs `provider=mock` unless a key is mounted.
