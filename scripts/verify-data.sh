#!/bin/bash
set -euo pipefail

# ── Verify Data Pipeline ──
# Post-deployment verification: check that data was fetched and stored correctly.
# Usage: ./verify-data.sh [ticker_group] [date]

STORAGE_ACCOUNT="tradingsystemsa"
CONTAINER="trading-data-blob"
SERVICE_BUS_NAMESPACE="trading-system-service-bus"

TICKER_GROUP="${1:-SP_500}"
CHECK_DATE="${2:-$(date -d yesterday +%Y-%m-%d 2>/dev/null || date -v-1d +%Y-%m-%d)}"

echo "=== Data Pipeline Verification ==="
echo "Ticker group: $TICKER_GROUP"
echo "Check date:   $CHECK_DATE"
echo ""

# ── 1. Check Blob Storage ──
echo "--- Blob Storage ---"
echo "Listing files for $TICKER_GROUP..."

# Calculate quarter for the check date
YEAR=$(echo "$CHECK_DATE" | cut -d'-' -f1)
MONTH=$(echo "$CHECK_DATE" | cut -d'-' -f2)
QUARTER_NUM=$(( (10#$MONTH - 1) / 3 + 1 ))
QUARTER="${YEAR}Q${QUARTER_NUM}"

PREFIX="${TICKER_GROUP}/${QUARTER}/"
echo "Looking in: $CONTAINER/$PREFIX"

FILES=$(az storage blob list \
    --account-name "$STORAGE_ACCOUNT" \
    --container-name "$CONTAINER" \
    --prefix "$PREFIX" \
    --auth-mode login \
    --query "[].{name:name, size:properties.contentLength, modified:properties.lastModified}" \
    -o table 2>/dev/null)

if [ -z "$FILES" ] || [ "$FILES" == "" ]; then
    echo "  WARNING: No files found!"
else
    echo "$FILES"
    FILE_COUNT=$(az storage blob list \
        --account-name "$STORAGE_ACCOUNT" \
        --container-name "$CONTAINER" \
        --prefix "$PREFIX" \
        --auth-mode login \
        --query "length([])" \
        -o tsv 2>/dev/null)
    echo ""
    echo "  Total files: $FILE_COUNT"
fi

echo ""

# ── 2. Check Service Bus ──
echo "--- Service Bus Topics ---"
for TOPIC in "raw-data-updates" "processed-data-updates" "model-creation" "model-updates"; do
    echo "Topic: $TOPIC"
    az servicebus topic show \
        --resource-group "trading-system-rg" \
        --namespace-name "$SERVICE_BUS_NAMESPACE" \
        --name "$TOPIC" \
        --query "{status:status, sizeInBytes:sizeInBytes, subscriptionCount:subscriptionCount}" \
        -o table 2>/dev/null || echo "  (not found)"

    # Show subscription message counts
    az servicebus topic subscription list \
        --resource-group "trading-system-rg" \
        --namespace-name "$SERVICE_BUS_NAMESPACE" \
        --topic-name "$TOPIC" \
        --query "[].{subscription:name, active:messageCount, deadLetter:deadLetterMessageCount}" \
        -o table 2>/dev/null || true
    echo ""
done

# ── 3. Check K8s Resources ──
echo "--- Kubernetes Resources ---"
NAMESPACE="trading"

echo "CronJob status:"
kubectl get cronjobs -n "$NAMESPACE" 2>/dev/null || echo "  (not available)"
echo ""

echo "Recent jobs:"
kubectl get jobs -n "$NAMESPACE" --sort-by=.metadata.creationTimestamp 2>/dev/null | tail -5 || echo "  (not available)"
echo ""

echo "Pod status:"
kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "  (not available)"
echo ""

# ── 4. Check IBKR Gateway ──
echo "--- IBKR Gateway ---"
GATEWAY_POD=$(kubectl get pods -n "$NAMESPACE" -l app=ibkr-gateway -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -n "$GATEWAY_POD" ]; then
    echo "Pod: $GATEWAY_POD"
    kubectl get pod "$GATEWAY_POD" -n "$NAMESPACE" -o jsonpath='{.status.phase}' 2>/dev/null
    echo ""
else
    echo "  Gateway pod not found"
fi

echo ""
echo "=== Verification complete ==="
