#!/usr/bin/env bash
set -euo pipefail

ISAACLAB_SH="${ISAACLAB_SH:-/home/khunanon/IsaacLab/isaaclab.sh}"
TASK="${TASK:-oneleg-uan-deploy}"
EVAL_SCRIPT="${EVAL_SCRIPT:-scripts/uan/evaluate_deploy_dataset.py}"
LOG_ROOT="${LOG_ROOT:-logs/rsl_rl/one_leg_uan_deploy}"
OUT_ROOT="${OUT_ROOT:-logs/rsl_rl/one_leg_uan_deploy/eval_drl_experiments}"
NUM_ENVS="${NUM_ENVS:-4096}"
STEPS="${STEPS:-1000}"
PLOT_STEPS="${PLOT_STEPS:-$STEPS}"
BATCHES="${BATCHES:-1}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-12}"
MODEL_NAME="${MODEL_NAME:-model_999.pt}"

WALK_AIR="real_data/uan_walk_air_20260505_155448.csv"
WALK_CONTACT="real_data/uan_walk_contact_20260505_165237.csv"
SQUARE="real_data/uan_square_20260505_145243.csv"
GAUSSIAN="real_data/uan_gaussian_20260505_145826.csv"
TRIANGLE="real_data/uan_test_triangle_20260505_174617.csv"
CHIRP="real_data/uan_test_chirp_20260505_174201.csv"

latest_run() {
    local run_name="$1"
    local run
    run="$(find "$LOG_ROOT" -maxdepth 1 -mindepth 1 -type d -name "*_${run_name}" | sort | tail -n 1)"
    if [[ -z "$run" ]]; then
        echo "[ERROR] Could not find run ending with: ${run_name}" >&2
        exit 1
    fi
    echo "$run"
}

evaluate_one() {
    local exp_name="$1"
    local case_name="$2"
    local run_name="$3"
    local history_len="$4"
    local dataset_path="$5"

    local run_dir
    run_dir="$(latest_run "$run_name")"
    local checkpoint="${run_dir}/${MODEL_NAME}"
    if [[ ! -f "$checkpoint" ]]; then
        echo "[ERROR] Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi

    local out_dir="${OUT_ROOT}/${exp_name}"
    mkdir -p "$out_dir"

    echo
    echo "[INFO] ${exp_name}/${case_name}"
    echo "[INFO]   run=${run_dir}"
    echo "[INFO]   history_len=${history_len}"
    echo "[INFO]   dataset=${dataset_path}"

    "$ISAACLAB_SH" -p "$EVAL_SCRIPT" \
        --task="$TASK" \
        --num_envs="$NUM_ENVS" \
        --steps="$STEPS" \
        --batches="$BATCHES" \
        --dataset_paths "$dataset_path" \
        --checkpoints "$checkpoint" \
        --history_len "$history_len" \
        --plot \
        --plot_steps="$PLOT_STEPS" \
        --episode_length_s="$EPISODE_LENGTH_S" \
        --output "${out_dir}/${case_name}_metrics.csv" \
        --plot_csv "${out_dir}/${case_name}_trace.csv" \
        --plot_output "${out_dir}/${case_name}.png" \
        --headless
}

echo "[INFO] Evaluating DRL experiments"
echo "[INFO] Results will be written to: ${OUT_ROOT}"

# Experiment 1: Observation Space Ablation.
evaluate_one "exp1_observation_ablation" "H1_walk_air" "exp1_H1" 1 "$WALK_AIR"
evaluate_one "exp1_observation_ablation" "H5_walk_air" "exp1_H5" 5 "$WALK_AIR"
evaluate_one "exp1_observation_ablation" "H10_walk_air" "exp1_H10" 10 "$WALK_AIR"
evaluate_one "exp1_observation_ablation" "H20_walk_air" "exp1_H20" 20 "$WALK_AIR"
evaluate_one "exp1_observation_ablation" "H40_walk_air" "exp1_H40" 40 "$WALK_AIR"

# Experiment 2: Data & Sample Efficiency.
evaluate_one "exp2_sample_efficiency" "p005_walk_air" "exp2_p005_H20" 20 "$WALK_AIR"
evaluate_one "exp2_sample_efficiency" "p010_walk_air" "exp2_p010_H20" 20 "$WALK_AIR"
evaluate_one "exp2_sample_efficiency" "p025_walk_air" "exp2_p025_H20" 20 "$WALK_AIR"
evaluate_one "exp2_sample_efficiency" "p050_walk_air" "exp2_p050_H20" 20 "$WALK_AIR"
evaluate_one "exp2_sample_efficiency" "p100_walk_air" "exp2_p100_H20" 20 "$WALK_AIR"

# Experiment 3: Sim-to-Real Gap Mitigation using the selected EXP1 H20 policy.
evaluate_one "exp3_sim_to_real_gap" "exp1_H20_walk_air" "exp1_H20" 20 "$WALK_AIR"

# Experiment 4: Surprise Test / OOD Generalization using the sine-only policy.
evaluate_one "exp4_ood_generalization" "square" "exp4_sine_only_H20" 20 "$SQUARE"
evaluate_one "exp4_ood_generalization" "gaussian" "exp4_sine_only_H20" 20 "$GAUSSIAN"
evaluate_one "exp4_ood_generalization" "triangle" "exp4_sine_only_H20" 20 "$TRIANGLE"
evaluate_one "exp4_ood_generalization" "chirp" "exp4_sine_only_H20" 20 "$CHIRP"
evaluate_one "exp4_ood_generalization" "walk_air" "exp4_sine_only_H20" 20 "$WALK_AIR"

# Experiment 5: Stress Test / Robustness using the selected EXP1 H20 policy.
evaluate_one "exp5_stress_test" "exp1_H20_walk_contact" "exp1_H20" 20 "$WALK_CONTACT"

echo
echo "[INFO] All evaluations finished."
