#!/usr/bin/env bash
# Download the API's packages for Linux into wheels/, so the Docker build can
# install them without using the network.
#
# Run it from the main project folder before building the image:
#   bash scripts/download_wheels.sh
#
# Why: on a slow or unstable connection, big downloads (scipy, numpy...) get
# cut off inside the Docker build, and every failed build starts over. Here,
# files that downloaded fine are kept, so each retry only fetches what's missing.

# Stop at the first error inside a command pipeline, and on unset variables.
set -u

# The container runs Linux on an ARM chip (aarch64) with Python 3.10.
# Packages are published for several Linux "manylinux" levels, so list them all.
PLATFORMS=(
  --platform manylinux2014_aarch64
  --platform manylinux_2_17_aarch64
  --platform manylinux_2_24_aarch64
  --platform manylinux_2_28_aarch64
)

for attempt in 1 2 3 4 5 6 7 8 9 10; do
  echo "--- Attempt $attempt"
  # pip download: fetch the files without installing them.
  #   --only-binary=:all:  ready-built wheels only (no building from source)
  #   --python-version 310 and the platforms: Linux files, not macOS ones
  #   -d wheels            save into the wheels/ folder
  # pip checks every file's fingerprint (hash) and skips ones already there.
  # setuptools and wheel are the build tools needed to install lendsmart itself.
  # greenlet is needed by sqlalchemy only on Linux ARM ("aarch64"). pip decides
  # that by looking at THIS Mac (whose chip it calls "arm64"), so it would skip
  # greenlet unless we ask for it by name.
  if .venv/bin/pip download \
      -r requirements-api.txt \
      "setuptools>=64" wheel greenlet \
      -d wheels \
      --only-binary=:all: \
      --python-version 310 \
      "${PLATFORMS[@]}" \
      --timeout 120 \
      --retries 10 \
      --quiet; then
    echo "All packages downloaded into wheels/"
    exit 0
  fi
  echo "A download failed; retrying (files already downloaded are kept)."
done

echo "Gave up after 10 attempts. Check the internet connection and run again."
exit 1
