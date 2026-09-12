// sec-filing-qa on Azure Container Apps (SPEC section 10).
//
// Scope: an existing resource group. Creates (or updates in place) a Log Analytics workspace, a
// Container Apps environment and one container app with external HTTPS ingress on port 8080,
// min 0 / max 2 replicas and 1 vCPU / 2 GiB per replica. The image is the public `full` target
// on GHCR, so no registry credential is needed. Secrets arrive as @secure() parameters and are
// stored as Container Apps secrets; an empty value leaves that variable unset.
//
// Deploy: az deployment group create -g <rg> --template-file main.bicep --parameters image=...

targetScope = 'resourceGroup'

@description('Container app name (also used for the environment and workspace names).')
@minLength(2)
@maxLength(32)
param name string = 'sec-filing-qa'

@description('Azure region for every resource.')
param location string = resourceGroup().location

@description('Fully qualified image reference, e.g. ghcr.io/<owner>/sec-filing-qa:<sha>.')
param image string

@description('Git commit exposed by GET /version (SECQA_GIT_SHA).')
param gitSha string = 'unknown'

@description('Index tarball / DuckDB URL downloaded at start; empty builds the fixture index.')
param indexUrl string = ''

@description('Embedder spec; must match the index (local for bge-small, hashing for the fixture).')
param embedder string = 'local'

@description('Default answering provider for anonymous callers.')
param provider string = 'mock'

@description('Per-request cost cap in USD.')
param maxCostUsd string = '0.25'

@description('Per-replica daily budget in USD (in-memory).')
param dailyBudgetUsd string = '5.0'

@description('vCPU per replica.')
param cpu string = '1.0'

@description('Memory per replica.')
param memory string = '2Gi'

@description('Minimum replicas (0 = scale to zero).')
@minValue(0)
@maxValue(10)
param minReplicas int = 0

@description('Maximum replicas.')
@minValue(1)
@maxValue(10)
param maxReplicas int = 2

@description('Concurrent requests per replica before scaling out.')
@minValue(1)
param concurrentRequests int = 4

@secure()
@description('OpenAI API key (empty = not configured).')
param openaiApiKey string = ''

@secure()
@description('Anthropic API key (empty = not configured).')
param anthropicApiKey string = ''

@secure()
@description('Demo API key gating agent mode and non-default providers (empty = open).')
param secqaApiKey string = ''

var containerPort = 8080

// Secrets: only the ones that were provided.
var secretDefs = concat(
  empty(openaiApiKey) ? [] : [{ name: 'openai-api-key', value: openaiApiKey }],
  empty(anthropicApiKey) ? [] : [{ name: 'anthropic-api-key', value: anthropicApiKey }],
  empty(secqaApiKey) ? [] : [{ name: 'secqa-api-key', value: secqaApiKey }]
)

var secretEnv = concat(
  empty(openaiApiKey) ? [] : [{ name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }],
  empty(anthropicApiKey) ? [] : [{ name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }],
  empty(secqaApiKey) ? [] : [{ name: 'SECQA_API_KEY', secretRef: 'secqa-api-key' }]
)

var plainEnv = concat(
  [
    { name: 'PORT', value: string(containerPort) }
    { name: 'SECQA_DUCKDB_PATH', value: '/data/index.duckdb' }
    { name: 'SECQA_EMBEDDER', value: embedder }
    { name: 'SECQA_PROVIDER', value: provider }
    { name: 'SECQA_MAX_COST_USD', value: maxCostUsd }
    { name: 'SECQA_DAILY_BUDGET_USD', value: dailyBudgetUsd }
    { name: 'SECQA_LOG_JSON', value: 'true' }
    { name: 'SECQA_GIT_SHA', value: gitSha }
  ],
  empty(indexUrl) ? [] : [{ name: 'SECQA_INDEX_URL', value: indexUrl }]
)

resource logs 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${name}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
    features: { enableLogAccessUsingOnlyResourcePermissions: true }
  }
}

resource env 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${name}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  properties: {
    managedEnvironmentId: env.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: containerPort
        transport: 'http'
        allowInsecure: false
        traffic: [{ latestRevision: true, weight: 100 }]
      }
      secrets: secretDefs
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          resources: {
            cpu: json(cpu)
            memory: memory
          }
          env: concat(plainEnv, secretEnv)
          probes: [
            {
              type: 'Startup'
              httpGet: { path: '/healthz', port: containerPort }
              initialDelaySeconds: 5
              periodSeconds: 5
              failureThreshold: 24
            }
            {
              type: 'Readiness'
              httpGet: { path: '/readyz', port: containerPort }
              periodSeconds: 10
              failureThreshold: 3
            }
            {
              type: 'Liveness'
              httpGet: { path: '/healthz', port: containerPort }
              periodSeconds: 30
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http'
            http: { metadata: { concurrentRequests: string(concurrentRequests) } }
          }
        ]
      }
    }
  }
}

@description('Public hostname of the app (https://<fqdn>).')
output fqdn string = app.properties.configuration.ingress.fqdn

@description('Resource id of the container app.')
output appId string = app.id
