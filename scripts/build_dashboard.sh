#!/bin/bash
set -e

echo "=========================================="
echo "  Building Kappa OS Dashboard"
echo "=========================================="

# Navigate to dashboard directory
cd "$(dirname "$0")/../src/dashboard"

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
npm install

# Build the dashboard
echo ""
echo "Building dashboard..."
npm run build

echo ""
echo "=========================================="
echo "  Dashboard built successfully!"
echo "=========================================="
echo ""
echo "Output: src/dashboard/dist/"
echo ""
echo "Start the server with:"
echo "  poetry run kappa dashboard"
echo ""
echo "Then open http://localhost:8000"
