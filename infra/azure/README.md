# Azure Container Apps deployment

`main.bicep` declares everything the app needs inside an existing resource group: a Log
Analytics workspace, a Container Apps environment and the container app itself (external HTTPS
ingress on port 8080, min 0 / max 2 replicas, 1 vCPU / 2 GiB). It pulls the `full` image from
GHCR **anonymously** (no `configuration.registries` credential), so the GHCR package must be
public; step 5 below makes it so, once. Vendor keys are `@secure()` parameters that become
Container Apps secrets; empty values leave the variable unset.

No local `az` CLI is assumed: the portal setup below is done once by hand, and
`.github/workflows/deploy-azure.yml` runs the Bicep deployment from CI with an OIDC federated
credential (no client secret anywhere).

## One-time setup (portal or Cloud Shell, Day 1)

1. Subscription and resource group, e.g. `rg-sec-filing-qa` in `eastus`.
2. Register the resource providers `Microsoft.App` and `Microsoft.OperationalInsights`.
3. App registration `sec-filing-qa-deployer`:
   - **Federated credential**: issuer `https://token.actions.githubusercontent.com`, subject
     `repo:<owner>/sec-filing-qa:environment:azure` (the workflow uses the `azure` environment),
     audience `api://AzureADTokenExchange`.
   - Role assignment **Contributor** on the resource group (Bicep creates the workspace,
     environment and app there).
4. GitHub repository settings:

| Kind     | Name                     | Value                                              |
|----------|--------------------------|----------------------------------------------------|
| variable | `DEPLOY_AZURE`           | `true`                                             |
| variable | `AZURE_CLIENT_ID`        | app registration (client) id                       |
| variable | `AZURE_TENANT_ID`        | directory (tenant) id                              |
| variable | `AZURE_SUBSCRIPTION_ID`  | subscription id                                    |
| variable | `AZURE_RESOURCE_GROUP`   | `rg-sec-filing-qa`                                 |
| variable | `AZURE_LOCATION`         | `eastus` (default)                                 |
| variable | `AZURE_APP_NAME`         | `sec-filing-qa` (default)                          |
| variable | `SECQA_INDEX_URL`        | published index tarball URL (empty = fixture index)|
| variable | `SECQA_EMBEDDER`         | `local` (published index) or `hashing` (fixture)   |
| secret   | `OPENAI_API_KEY` etc.    | optional; empty = not configured on the app        |

5. Make the GHCR package public, after the first `build.yml` run. `build.yml` pushes with
   `secrets.GITHUB_TOKEN`, and a package created that way is **private** by default, so
   Container Apps (which pulls with no credential) would fail to provision the revision with an
   image-pull authorization error. On GitHub: your profile (or organisation) → **Packages** →
   `sec-filing-qa` → **Package settings** → **Danger zone** → **Change visibility** → **Public**.
   Both the `full` and `-slim` tags live in that one package, so this is a single switch. The
   deploy workflow verifies the anonymous pull path before it touches Azure and fails with a
   pointer to this step if the package is still private.

## Deploying

Push a `v*` tag or dispatch `deploy-azure` manually. The job runs

```bash
az deployment group create --resource-group rg-sec-filing-qa \
    --template-file infra/azure/main.bicep \
    --parameters name=sec-filing-qa image=ghcr.io/<owner>/sec-filing-qa:<sha> ...
```

then curls `/readyz` and `POST /v1/ask` (`provider=mock`) on the app's FQDN and writes both JSON
bodies into the job summary. Until that job has one green run, the README must say
"workflow written, deployment not yet verified" (SPEC section 9).

## Notes

- Container Apps consumption plan bills per vCPU-second while replicas run; `minReplicas: 0`
  scales to zero after idle and the first request pays the ~15 s cold start of the full image.
- The `PORT` environment variable is set to 8080 explicitly; the entrypoint honours it.
- To change CPU/memory or replica bounds edit the parameters in `main.bicep` (they are
  parameters, not literals, so CI can override them without a template change).
