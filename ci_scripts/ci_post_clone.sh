#!/bin/sh
set -eo pipefail

WORKSPACE_PATH="${CI_WORKSPACE:-${CI_PRIMARY_REPOSITORY_PATH:-$(pwd)}}"
echo "[ci_post_clone] workspace: ${WORKSPACE_PATH}"
cd "${WORKSPACE_PATH}"

if [ -z "${ROBOFLOW_API_KEY:-}" ]; then
  echo "[ci_post_clone] ERROR: ROBOFLOW_API_KEY is not set in Xcode Cloud workflow variables."
  exit 1
fi

ROBOFLOW_PILL_MODEL_ID="${ROBOFLOW_PILL_MODEL_ID:-pill_count-instance-segment}"
ROBOFLOW_PILL_MODEL_VERSION="${ROBOFLOW_PILL_MODEL_VERSION:-8}"

cat > Secrets.xcconfig <<EOT
ROBOFLOW_API_KEY = ${ROBOFLOW_API_KEY}
ROBOFLOW_PILL_MODEL_ID = ${ROBOFLOW_PILL_MODEL_ID}
ROBOFLOW_PILL_MODEL_VERSION = ${ROBOFLOW_PILL_MODEL_VERSION}
EOT
echo "[ci_post_clone] wrote Secrets.xcconfig"

if [ -f "Gemfile" ] && command -v bundle >/dev/null 2>&1; then
  echo "[ci_post_clone] running bundle exec pod install"
  bundle exec pod install
else
  if ! command -v pod >/dev/null 2>&1; then
    echo "[ci_post_clone] ERROR: CocoaPods not found (pod command missing)."
    exit 1
  fi
  echo "[ci_post_clone] running pod install"
  pod install
fi
