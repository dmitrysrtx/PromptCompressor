#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Default parameters optimized for A100 / L4
EVAL_TYPE=""
GEN_MODEL="gpt2-xl-code"
PCRL_MODEL="roberta_code_v1_bleu_512_part2"
SEED=42
BS_ORIG=64
BS_PCRL=32

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--type) EVAL_TYPE="$2"; shift ;;
        -m|--gen_model) GEN_MODEL="$2"; shift ;;
        -p|--pcrl_model) PCRL_MODEL="$2"; shift ;;
        -b|--bs) 
            BS_ORIG="$2"
            BS_PCRL="$2"
            shift ;;
        -h|--help)
            echo "Usage: ./run_eval.sh -t [original|pcrl] [options]"
            echo "Options:"
            echo "  -t, --type         Type of evaluation (required: original or pcrl)"
            echo "  -b, --bs           Batch size override"
            echo "  -p, --pcrl_model   PCRL model directory name"
            exit 0
            ;;
        *) echo "[ERROR] Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Validate that the type was provided
if [ -z "$EVAL_TYPE" ]; then
    echo "[ERROR] Evaluation type is required."
    echo "Please use: ./run_eval.sh -t original OR ./run_eval.sh -t pcrl"
    exit 1
fi

echo "=================================================="
echo "[INFO] Starting Evaluation Pipeline"
echo "[INFO] Type: $EVAL_TYPE"
echo "=================================================="

# Activate environment and run the requested script
if [ "$EVAL_TYPE" == "original" ]; then
    
    echo "[INFO] Running Original Evaluation with Gen Model: $GEN_MODEL, Batch Size: $BS_ORIG"
    source setup_env.sh && /content/env39/bin/python evaluate_original.py \
        --gen_model "$GEN_MODEL" \
        --bs "$BS_ORIG"

elif [ "$EVAL_TYPE" == "pcrl" ]; then
    
    echo "[INFO] Running PCRL Evaluation with PCRL Model: $PCRL_MODEL, Seed: $SEED, Batch Size: $BS_PCRL"
    source setup_env.sh && /content/env39/bin/python evaluate_pcrl.py \
        --pcrl_model "$PCRL_MODEL" \
        --gen_model "$GEN_MODEL" \
        --seed "$SEED" \
        --bs "$BS_PCRL"

else
    echo "[ERROR] Invalid type: $EVAL_TYPE. Use 'original' or 'pcrl'."
    exit 1
fi

echo "=================================================="
echo "[INFO] Evaluation completed successfully!"
echo "=================================================="