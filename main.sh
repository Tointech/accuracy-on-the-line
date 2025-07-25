#!/bin/bash

NUM_RUNS=10
OUT_FILE="cifar_exp_results.json"
FIG_NAME="cifar_exp_plot.png"

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
