#!/bin/bash -e
# Docker wrapper for Hexagon builds using llama.cpp's toolchain image
# This ensures consistent builds with all dependencies pre-installed

DOCKER_IMAGE="ghcr.io/snapdragon-toolchain/arm64-android:v0.3"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Docker Build Wrapper for Hexagon"
echo "=========================================="
echo "Image: $DOCKER_IMAGE"
echo "Project: $PROJECT_ROOT"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
    echo ""
    echo "Please install Docker:"
    echo "  - macOS: https://docs.docker.com/desktop/install/mac-install/"
    echo "  - Linux: https://docs.docker.com/engine/install/"
    echo "  - Windows: https://docs.docker.com/desktop/install/windows-install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    echo -e "${RED}Error: Docker daemon is not running${NC}"
    echo ""
    echo "Please start Docker Desktop or the Docker daemon"
    exit 1
fi

# Determine platform based on host
PLATFORM="linux/amd64"

# Pull the image if not present
echo "Checking Docker image..."
if ! docker image inspect "$DOCKER_IMAGE" &> /dev/null; then
    echo -e "${YELLOW}Docker image not found locally. Pulling...${NC}"
    docker pull --platform "$PLATFORM" "$DOCKER_IMAGE"
    echo -e "${GREEN}✓ Image pulled successfully${NC}"
else
    echo -e "${GREEN}✓ Image already present${NC}"
fi
echo ""

# Run command in Docker
# Usage: ./docker-build-wrapper.sh <command>
COMMAND="${1:-bash}"

echo "Running in Docker container..."
echo "Command: $COMMAND"
echo ""

# Run the container
docker run --rm \
    -u "$(id -u):$(id -g)" \
    --volume "$PROJECT_ROOT:/workspace" \
    --workdir /workspace \
    --platform "$PLATFORM" \
    "$DOCKER_IMAGE" \
    bash -c "$COMMAND"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Docker build completed successfully${NC}"
else
    echo ""
    echo -e "${RED}✗ Docker build failed with exit code $EXIT_CODE${NC}"
fi

exit $EXIT_CODE
