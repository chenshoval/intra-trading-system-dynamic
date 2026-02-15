#!/bin/bash
set -euo pipefail

# ── Setup Azure Service Bus Topics + Subscriptions ──
# Migrated from queues to topics for fan-out support.
# Usage: ./setup-service-bus.sh [create|delete|status]

RESOURCE_GROUP="trading-system-rg"
NAMESPACE_NAME="trading-system-service-bus"

ACTION="${1:-create}"

echo "=== Service Bus Setup ==="
echo "Namespace: $NAMESPACE_NAME"
echo "Action:    $ACTION"
echo ""

# Topic + subscription definitions
# Format: topic_name:subscription1,subscription2
TOPICS=(
    "raw-data-updates:feature-engine-sub"
    "processed-data-updates:model-trainer-sub"
    "model-creation:model-evaluator-sub"
    "model-updates:execution-engine-sub"
)

create_topic() {
    local topic_name=$1
    echo "Creating topic: $topic_name..."
    az servicebus topic create \
        --resource-group "$RESOURCE_GROUP" \
        --namespace-name "$NAMESPACE_NAME" \
        --name "$topic_name" \
        --default-message-time-to-live PT2H \
        --max-size 1024 \
        --enable-partitioning true \
        2>/dev/null && echo "  Topic '$topic_name' created." || echo "  Topic '$topic_name' already exists."
}

create_subscription() {
    local topic_name=$1
    local sub_name=$2
    echo "  Creating subscription: $sub_name on $topic_name..."
    az servicebus topic subscription create \
        --resource-group "$RESOURCE_GROUP" \
        --namespace-name "$NAMESPACE_NAME" \
        --topic-name "$topic_name" \
        --name "$sub_name" \
        --max-delivery-count 10 \
        --lock-duration PT30S \
        --default-message-time-to-live PT2H \
        2>/dev/null && echo "    Subscription '$sub_name' created." || echo "    Subscription '$sub_name' already exists."
}

delete_topic() {
    local topic_name=$1
    echo "Deleting topic: $topic_name..."
    az servicebus topic delete \
        --resource-group "$RESOURCE_GROUP" \
        --namespace-name "$NAMESPACE_NAME" \
        --name "$topic_name" \
        2>/dev/null && echo "  Deleted." || echo "  Not found."
}

# Also delete old queues if they exist
delete_old_queues() {
    echo ""
    echo "Cleaning up old queues (migrated to topics)..."
    for queue in "raw-data-updates" "processed-data-updates" "model-creation" "model-updates"; do
        echo "  Deleting queue: $queue..."
        az servicebus queue delete \
            --resource-group "$RESOURCE_GROUP" \
            --namespace-name "$NAMESPACE_NAME" \
            --name "$queue" \
            2>/dev/null && echo "    Deleted." || echo "    Not found (already cleaned up)."
    done
}

show_status() {
    echo "Topics:"
    az servicebus topic list \
        --resource-group "$RESOURCE_GROUP" \
        --namespace-name "$NAMESPACE_NAME" \
        --query "[].{name:name, status:status, sizeInBytes:sizeInBytes}" \
        -o table 2>/dev/null || echo "  (none)"

    echo ""
    for entry in "${TOPICS[@]}"; do
        IFS=':' read -r topic_name subs <<< "$entry"
        echo "Subscriptions for '$topic_name':"
        az servicebus topic subscription list \
            --resource-group "$RESOURCE_GROUP" \
            --namespace-name "$NAMESPACE_NAME" \
            --topic-name "$topic_name" \
            --query "[].{name:name, messageCount:messageCount, status:status}" \
            -o table 2>/dev/null || echo "  (none)"
    done

    echo ""
    echo "Queues (legacy — should be empty after migration):"
    az servicebus queue list \
        --resource-group "$RESOURCE_GROUP" \
        --namespace-name "$NAMESPACE_NAME" \
        --query "[].{name:name, messageCount:messageCountDetails.activeMessageCount, status:status}" \
        -o table 2>/dev/null || echo "  (none)"
}

case "$ACTION" in
    create)
        for entry in "${TOPICS[@]}"; do
            IFS=':' read -r topic_name subs <<< "$entry"
            create_topic "$topic_name"
            IFS=',' read -ra sub_array <<< "$subs"
            for sub in "${sub_array[@]}"; do
                create_subscription "$topic_name" "$sub"
            done
            echo ""
        done
        echo "=== Service Bus topics + subscriptions created ==="
        ;;

    migrate)
        # Full migration: create topics, then delete old queues
        echo "Running full migration: queues -> topics..."
        echo ""
        for entry in "${TOPICS[@]}"; do
            IFS=':' read -r topic_name subs <<< "$entry"
            create_topic "$topic_name"
            IFS=',' read -ra sub_array <<< "$subs"
            for sub in "${sub_array[@]}"; do
                create_subscription "$topic_name" "$sub"
            done
            echo ""
        done
        delete_old_queues
        echo ""
        echo "=== Migration complete ==="
        ;;

    delete)
        for entry in "${TOPICS[@]}"; do
            IFS=':' read -r topic_name _ <<< "$entry"
            delete_topic "$topic_name"
        done
        echo "=== Topics deleted ==="
        ;;

    status)
        show_status
        ;;

    *)
        echo "Usage: $0 [create|migrate|delete|status]"
        echo ""
        echo "  create  - Create topics + subscriptions"
        echo "  migrate - Create topics + delete old queues"
        echo "  delete  - Delete all topics"
        echo "  status  - Show current state"
        exit 1
        ;;
esac
