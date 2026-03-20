#!/usr/bin/env bash
set -euo pipefail

# Clone dependencies if not already present
[ ! -d kronfluence ] && git clone https://github.com/pomonam/kronfluence.git
[ ! -d bergson ]     && git clone https://github.com/EleutherAI/bergson.git

# Apply patches
cd kronfluence && git apply ../patches/kronfluence.patch && cd ..
cd bergson && git checkout dev && cd ..

# Install
uv pip install -e ./bergson
uv pip install -e ./kronfluence
uv pip install -e .
