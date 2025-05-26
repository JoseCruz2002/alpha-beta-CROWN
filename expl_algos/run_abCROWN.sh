#!/bin/bash

CONFIG_FILE_PATH=$1
OUTPUT_FILE_PATH=$2

cd /home/josecruz/Documents/MEIC/Thesis/alpha-beta-CROWN/complete_verifier/
python abcrown.py --config "$CONFIG_FILE_PATH" > "$OUTPUT_FILE_PATH"