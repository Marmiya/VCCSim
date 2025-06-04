#!/bin/bash

# Install required packages
pip install -r requirements.txt

# Set paths
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_PATH="${CURRENT_DIR}/../../Source/pb/VCCSim.proto"
OUTPUT_PATH="${CURRENT_DIR}/VCCSim"

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_PATH"

# Generate VCCSim_pb2.py and VCCSim_pb2_grpc.py and copy them to the GRPC folder
python -m grpc_tools.protoc -I "${CURRENT_DIR}/../../Source/pb" --python_out="$OUTPUT_PATH" --grpc_python_out="$OUTPUT_PATH" "$PROTO_PATH"

# Fix the import statement in VCCSim_pb2_grpc.py
GRPC_FILE="${OUTPUT_PATH}/VCCSim_pb2_grpc.py"
sed -i "s/import VCCSim_pb2 as VCCSim__pb2/from . import VCCSim_pb2 as VCCSim__pb2/g" "$GRPC_FILE"

echo "Proto generation completed with import fixes."