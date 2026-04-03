# Outputs Directory

Generated experiment artifacts and local validation results are written here.

Common locations:
- `outputs/marl_train_validation/` for dataset-backed MARL training and validation
- `outputs/<run_name>/` for experiment harness summaries and plots
- `outputs/<run_name>/interactive_*.html` for interactive Plotly charts
- `outputs/*_simulation_log.csv` for optional demo policy logs

These files are ignored by git because they are reproducible local outputs.
