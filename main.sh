#!/bin/bash

NUM_RUNS=2
OUT_FILE="test.json"
FIG_NAME="test.png"

# Step 1: Run the experiment
echo "Running experiments..."
python src/run_experiment.py \
    --num_runs $NUM_RUNS \
    --output_path $OUT_FILE

# Step 2: Plot the results
echo "Generating plot..."
python src/plot.py \
    --results_path "models/outputs/$OUT_FILE" \
    --save_name "$FIG_NAME"
