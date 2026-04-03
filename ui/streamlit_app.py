from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.run_collision_avoidance_tests import build_test_cases
from sim.reporting import (
    RUN_METRICS,
    SUMMARY_METRICS,
    build_pareto_figure,
    build_run_distribution_figure,
    build_summary_bar_figure,
    build_training_progress_figure,
)

PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
TRAINED_MODEL = ROOT / "outputs" / "marl_train_validation" / "marl_trained_from_train_dataset.pth"


def run_command(args: List[str]) -> subprocess.CompletedProcess[str]:
    command = [str(PYTHON_EXE), *args]
    return subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def output_dirs_with_results() -> List[Path]:
    outputs_root = ROOT / "outputs"
    if not outputs_root.exists():
        return []

    candidates = set()
    marker_files = {
        "aggregated_summary.csv",
        "validation_policy_summary.csv",
        "test_runs_per_policy.csv",
        "validation_episode_metrics.csv",
        "train_metrics.csv",
    }
    for path in outputs_root.rglob("*"):
        if path.is_file() and path.name in marker_files:
            candidates.add(path.parent)
    return sorted(candidates)


def show_command_result(result: subprocess.CompletedProcess[str]) -> None:
    st.code(" ".join(result.args), language="powershell")
    if result.stdout:
        st.text_area("stdout", result.stdout, height=280)
    if result.stderr:
        st.text_area("stderr", result.stderr, height=180)
    if result.returncode == 0:
        st.success("Command completed successfully.")
    else:
        st.error(f"Command failed with exit code {result.returncode}.")


def render_summary_charts(summary_df: pd.DataFrame) -> None:
    st.subheader("Summary Table")
    st.dataframe(summary_df, use_container_width=True)

    for metric, label in SUMMARY_METRICS.items():
        if metric not in summary_df.columns:
            continue
        fig = build_summary_bar_figure(summary_df, metric)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)

    pareto = build_pareto_figure(summary_df)
    if pareto is not None:
        st.plotly_chart(pareto, use_container_width=True)


def render_runs_charts(runs_df: pd.DataFrame) -> None:
    st.subheader("Raw Episode Runs")
    st.dataframe(runs_df, use_container_width=True)

    for metric, label in RUN_METRICS.items():
        if metric not in runs_df.columns:
            continue
        fig = build_run_distribution_figure(runs_df, metric)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


def render_training_charts(train_df: pd.DataFrame) -> None:
    st.subheader("Training Metrics")
    st.dataframe(train_df, use_container_width=True)

    for metric in ["final_collisions", "final_fuel_used", "final_steps", "actor_loss", "critic_loss"]:
        if metric not in train_df.columns:
            continue
        fig = build_training_progress_figure(train_df, metric)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True)


st.set_page_config(page_title="FuelSafe Simulator UI", layout="wide")
st.title("FuelSafe Simulator UI")
st.caption("Run demos, experiments, train/validate MARL, and inspect policy comparisons.")

page = st.sidebar.radio("Mode", ["Run Scenarios", "Explore Results"], index=0)

if page == "Run Scenarios":
    tabs = st.tabs(
        [
            "Demo",
            "Dataset Experiment",
            "Train + Validate",
            "Test Framework",
            "Advanced Example",
            "CSV Stats",
        ]
    )

    with tabs[0]:
        st.subheader("Run Demo Comparison")
        with st.form("demo_form"):
            episodes = st.number_input("Episodes", min_value=1, value=3)
            steps = st.number_input("Steps", min_value=1, value=500)
            sats = st.number_input("Satellites", min_value=1, value=3)
            debris = st.number_input("Debris", min_value=1, value=5)
            include_marl = st.checkbox("Include trained MARL", value=TRAINED_MODEL.exists())
            submit = st.form_submit_button("Run Demo")

        if submit:
            args = [
                "main.py",
                "--demo",
                "--episodes",
                str(episodes),
                "--steps",
                str(steps),
                "--sats",
                str(sats),
                "--debris",
                str(debris),
            ]
            if include_marl:
                args.extend(["--include-marl", "--marl-model-path", str(TRAINED_MODEL)])
            with st.spinner("Running demo..."):
                show_command_result(run_command(args))

    with tabs[1]:
        st.subheader("Run Dataset Experiment")
        with st.form("experiment_form"):
            dataset_path = st.text_input("Dataset CSV", str(ROOT / "data" / "test_data.csv"))
            steps = st.number_input("Steps per scenario", min_value=1, value=120, key="exp_steps")
            sats = st.number_input("Satellites", min_value=1, value=3, key="exp_sats")
            debris = st.number_input("Debris", min_value=1, value=10, key="exp_debris")
            max_scenarios = st.number_input("Max scenarios", min_value=1, value=8)
            include_marl = st.checkbox("Include trained MARL", value=TRAINED_MODEL.exists(), key="exp_marl")
            submit = st.form_submit_button("Run Experiment")

        if submit:
            args = [
                "main.py",
                "--experiment",
                "--dataset",
                dataset_path,
                "--steps",
                str(steps),
                "--sats",
                str(sats),
                "--debris",
                str(debris),
                "--max-scenarios",
                str(max_scenarios),
                "--risk-threshold",
                "-7.0",
            ]
            if include_marl:
                args.extend(["--include-marl", "--marl-model-path", str(TRAINED_MODEL)])
            with st.spinner("Running experiment..."):
                show_command_result(run_command(args))

    with tabs[2]:
        st.subheader("Train And Validate MARL")
        with st.form("train_validate_form"):
            train_csv = st.text_input("Train CSV", str(ROOT / "data" / "train_data.csv"))
            test_csv = st.text_input("Test CSV", str(ROOT / "data" / "test_data.csv"))
            output_dir = st.text_input("Output directory", str(ROOT / "outputs" / "ui" / "marl_train_validation"))
            train_rows = st.number_input("Train max rows", min_value=1, value=5000)
            test_rows = st.number_input("Test max rows", min_value=1, value=2000)
            train_scenarios = st.number_input("Train scenarios", min_value=1, value=12)
            test_scenarios = st.number_input("Test scenarios", min_value=1, value=8)
            episodes_per_scenario = st.number_input("Episodes per scenario", min_value=1, value=4)
            max_steps = st.number_input("Max steps", min_value=1, value=120)
            sats = st.number_input("Satellites", min_value=1, value=3, key="tv_sats")
            debris = st.number_input("Debris", min_value=1, value=10, key="tv_debris")
            epochs_per_batch = st.number_input("MARL epochs per batch", min_value=1, value=3)
            submit = st.form_submit_button("Train + Validate")

        if submit:
            args = [
                "sim/dataset_integration.py",
                "--train-csv",
                train_csv,
                "--test-csv",
                test_csv,
                "--output-dir",
                output_dir,
                "--train-max-rows",
                str(train_rows),
                "--test-max-rows",
                str(test_rows),
                "--risk-threshold",
                "-7.0",
                "--train-scenarios",
                str(train_scenarios),
                "--test-scenarios",
                str(test_scenarios),
                "--episodes-per-scenario",
                str(episodes_per_scenario),
                "--max-steps",
                str(max_steps),
                "--num-satellites",
                str(sats),
                "--num-debris",
                str(debris),
                "--marl-epochs-per-batch",
                str(epochs_per_batch),
            ]
            with st.spinner("Training and validating MARL..."):
                show_command_result(run_command(args))

    with tabs[3]:
        st.subheader("Run Test Framework")
        scenario_keys = list(build_test_cases(max_debris=200).keys())
        with st.form("test_framework_form"):
            selected_cases = st.multiselect("Test cases", scenario_keys, default=scenario_keys)
            mc_runs = st.number_input("Monte Carlo runs", min_value=1, value=1)
            max_debris = st.number_input("Max debris cap", min_value=1, value=200)
            output_dir = st.text_input("Output directory", str(ROOT / "outputs" / "ui" / "test_framework"))
            include_marl = st.checkbox("Include MARL", value=TRAINED_MODEL.exists(), key="tf_marl")
            use_trained = st.checkbox("Use trained MARL weights", value=TRAINED_MODEL.exists())
            quick = st.checkbox("Quick mode", value=False)
            submit = st.form_submit_button("Run Tests")

        if submit:
            args = [
                "experiments/run_collision_avoidance_tests.py",
                "--mc-runs",
                str(mc_runs),
                "--max-debris",
                str(max_debris),
                "--output-dir",
                output_dir,
            ]
            if selected_cases and len(selected_cases) != len(scenario_keys):
                args.extend(["--test-cases", ",".join(selected_cases)])
            if quick:
                args.append("--quick")
            if include_marl:
                args.append("--include-marl")
                if use_trained and TRAINED_MODEL.exists():
                    args.extend(["--marl-model-path", str(TRAINED_MODEL)])
                else:
                    args.append("--marl-untrained")
            with st.spinner("Running test framework..."):
                show_command_result(run_command(args))

    with tabs[4]:
        st.subheader("Advanced Example")
        if st.button("Run advanced_example.py"):
            with st.spinner("Running advanced examples..."):
                show_command_result(run_command(["advanced_example.py"]))

    with tabs[5]:
        st.subheader("CSV Data Loader Statistics")
        if st.button("Run sim/csv_data_loader.py"):
            with st.spinner("Loading dataset stats..."):
                show_command_result(run_command(["sim/csv_data_loader.py"]))

else:
    st.subheader("Explore Existing Outputs")
    available_dirs = output_dirs_with_results()
    if not available_dirs:
        st.info("No result directories found under outputs/.")
    else:
        selected_dir = st.selectbox(
            "Result directory",
            available_dirs,
            format_func=lambda p: str(p.relative_to(ROOT)),
        )

        summary_path = selected_dir / "aggregated_summary.csv"
        runs_path = selected_dir / "test_runs_per_policy.csv"
        validation_summary_path = selected_dir / "validation_policy_summary.csv"
        validation_runs_path = selected_dir / "validation_episode_metrics.csv"
        train_metrics_path = selected_dir / "train_metrics.csv"

        if summary_path.exists():
            render_summary_charts(pd.read_csv(summary_path))
        if validation_summary_path.exists():
            render_summary_charts(pd.read_csv(validation_summary_path))
        if runs_path.exists():
            render_runs_charts(pd.read_csv(runs_path))
        if validation_runs_path.exists():
            render_runs_charts(pd.read_csv(validation_runs_path))
        if train_metrics_path.exists():
            render_training_charts(pd.read_csv(train_metrics_path))
