#!/bin/bash
###############################################################################
# AI:OS Azure Automated Deployment Script
###############################################################################

set -e

RESOURCE_GROUP="${RESOURCE_GROUP:-aios-rg}"
LOCATION="${LOCATION:-eastus}"
CONTAINER_APP_ENV="${CONTAINER_APP_ENV:-aios-env}"
APP_NAME="${APP_NAME:-aios-prime}"
REGISTRY_NAME="${REGISTRY_NAME:-aiosregistry$(date +%s)}"

echo "Logging in to Azure... (Ensure AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET are set or use Managed Identity)"
# Assuming az login is handled outside or via env vars for service principals

echo "Creating Resource Group $RESOURCE_GROUP in $LOCATION..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "Creating Azure Container Registry $REGISTRY_NAME..."
az acr create --resource-group "$RESOURCE_GROUP" --name "$REGISTRY_NAME" --sku Basic --admin-enabled true

echo "Building Docker image and pushing to ACR..."
# Azure Container Registry build command builds and pushes directly in the cloud
az acr build --registry "$REGISTRY_NAME" --image "aios-runtime:latest" -f Dockerfile .

echo "Getting ACR credentials..."
ACR_LOGIN_SERVER=$(az acr show --name "$REGISTRY_NAME" --query loginServer --output tsv)
ACR_USERNAME=$(az acr credential show --name "$REGISTRY_NAME" --query username --output tsv)
ACR_PASSWORD=$(az acr credential show --name "$REGISTRY_NAME" --query "passwords[0].value" --output tsv)

echo "Creating Azure Container Apps Environment $CONTAINER_APP_ENV..."
az containerapp env create \
    --name "$CONTAINER_APP_ENV" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION"

echo "Deploying AI:OS Container App..."
az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINER_APP_ENV" \
    --image "$ACR_LOGIN_SERVER/aios-runtime:latest" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USERNAME" \
    --registry-password "$ACR_PASSWORD" \
    --target-port 8080 \
    --ingress external \
    --env-vars PORT=8080 LOG_LEVEL=info ALLOW_NETWORK_CALLS=true AIOS_LAUNCHER_PORT=7777 \
    --cpu 2.0 --memory 4.0Gi

echo "Deployment complete! Fetching FQDN..."
FQDN=$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv)

echo "AI:OS is live at: https://$FQDN"
echo "API endpoints are exposed on this URL (e.g. https://$FQDN/api/status, https://$FQDN/api/echo_prime)"
