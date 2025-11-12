#!/bin/bash
# Helper script to set up Hexagon SDK environment variables
# Source this file in your shell: source scripts/setup-hexagon-env.sh

HEXAGON_SDK_VERSION="6.4.0.2"
HEXAGON_TOOLS_VERSION="19.0.04"
HEXAGON_INSTALL_DIR="${HEXAGON_INSTALL_DIR:-$HOME/.hexagon-sdk}"

if [ ! -d "$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION" ]; then
  echo "Error: Hexagon SDK not found at $HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
  echo "Please run 'npm run bootstrap' to download and install the SDK"
  return 1
fi

export HEXAGON_SDK_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION"
export HEXAGON_TOOLS_ROOT="$HEXAGON_INSTALL_DIR/$HEXAGON_SDK_VERSION/tools/HEXAGON_Tools/$HEXAGON_TOOLS_VERSION"
export DEFAULT_HLOS_ARCH=64
export DEFAULT_TOOLS_VARIANT=toolv19
export DEFAULT_NO_QURT_INC=0
export DEFAULT_DSP_ARCH=v73

echo "Hexagon SDK environment variables set:"
echo "  HEXAGON_SDK_ROOT=$HEXAGON_SDK_ROOT"
echo "  HEXAGON_TOOLS_ROOT=$HEXAGON_TOOLS_ROOT"
echo ""
echo "You can now build with Hexagon backend support."
