#!/bin/bash

# Stop and cleanup script for ML Server Demo

echo "🛑 Stopping ML Server Demo..."

# Stop services
docker-compose down

echo "🧹 Cleaning up (optional - uncomment if needed)..."
# Uncomment the following lines if you want to clean up images and volumes
# echo "Removing images..."
# docker-compose down --rmi all
# echo "Removing volumes..."
# docker-compose down --volumes

echo "✅ Cleanup complete!"
echo ""
echo "💡 To restart services, run: ./deploy.sh"