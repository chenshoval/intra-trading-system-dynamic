#!/bin/bash
set -euo pipefail

# ── Train Model on Azure (ACI) ──
# Submits a training job to Azure Container Instances.
# The model is saved to Azure Blob Storage after training.
#
# Usage:
#   ./scripts/train-on-azure.sh [data_blob_path]
#
# Prerequisites:
#   - ACR image built: docker build -t tradingsystemacr.azurecr.io/training:latest -f training/Dockerfile .
#   - ACR image pushed: docker push tradingsystemacr.azurecr.io/training:latest
#   - Azure CLI logged in

REGISTRY="tradingsystemacr.azurecr.io"
IMAGE="training"
TAG="${TAG:-latest}"
RESOURCE_GROUP="trading-system-rg"
CONTAINER_NAME="training-job-$(date +%Y%m%d-%H%M%S)"
DATA_PATH="${1:-data/processed/features.csv}"

echo "=== Training Job ==="
echo "Image:     $REGISTRY/$IMAGE:$TAG"
echo "Data:      $DATA_PATH"
echo "Container: $CONTAINER_NAME"
echo ""

# Build and push if needed
read -p "Build and push image? (y/N): " BUILD
if [[ "$BUILD" == "y" || "$BUILD" == "Y" ]]; then
    echo "Building..."
    docker build -t "$REGISTRY/$IMAGE:$TAG" -f training/Dockerfile .

    echo "Pushing..."
    az acr login --name tradingsystemacr
    docker push "$REGISTRY/$IMAGE:$TAG"
fi

# Get ACR credentials
ACR_USER=$(az acr credential show --name tradingsystemacr --query username -o tsv)
ACR_PASS=$(az acr credential show --name tradingsystemacr --query "passwords[0].value" -o tsv)

# Create ACI container
echo "Submitting training job..."
az container create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$CONTAINER_NAME" \
    --image "$REGISTRY/$IMAGE:$TAG" \
    --registry-login-server "$REGISTRY" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --cpu 4 \
    --memory 8 \
    --restart-policy Never \
    --command-line "python training/train_lightgbm.py --data $DATA_PATH --upload-azure" \
    --environment-variables \
        ACCOUNT_URL="https://tradingsystemsa.blob.core.windows.net" \
    --assign-identity "/subscriptions/$(az account show --query id -o tsv)/resourcegroups/$RESOURCE_GROUP/providers/Microsoft.ManagedIdentity/userAssignedIdentities/trading-system-mi" \
    -o table

echo ""
echo "Job submitted: $CONTAINER_NAME"
echo ""
echo "Monitor:"
echo "  az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --follow"
echo ""
echo "Cleanup after completion:"
echo "  az container delete --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --yes"
