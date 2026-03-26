import pandas as pd
import matplotlib.pyplot as plt


def plot_simulation_stats(log_df, title="Simulation Log"):
    """Plot collisions/alerts and fuel usage over steps."""
    required = {'timesteps', 'collisions', 'alerts', 'fuel_used'}
    if not required.issubset(set(log_df.columns)):
        raise ValueError(f"Missing columns: {required - set(log_df.columns)}")

    plt.figure(figsize=(12, 6))
    plt.plot(log_df['timesteps'], log_df['collisions'], label='Collisions', color='tab:red', marker='o')
    plt.plot(log_df['timesteps'], log_df['alerts'], label='Alerts', color='tab:orange', marker='x')
    plt.xlabel('Step')
    plt.ylabel('Count')
    plt.title(f"{title}: Collisions & Alerts")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(log_df['timesteps'], log_df['fuel_used'], label='Fuel used (kg)', color='tab:blue', marker='s')
    plt.xlabel('Step')
    plt.ylabel('Fuel used (kg)')
    plt.title(f"{title}: Fuel Used")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_policies(policy_logs, metric='fuel_used'):
    """Compare metric across two or more policy logs."""
    if metric not in {'fuel_used', 'collisions', 'alerts'}:
        raise ValueError("Metric must be one of 'fuel_used', 'collisions', 'alerts'")

    plt.figure(figsize=(12, 6))
    for policy_name, df in policy_logs.items():
        if 'timesteps' not in df.columns or metric not in df.columns:
            raise ValueError(f"Policy {policy_name} missing required columns")
        plt.plot(df['timesteps'], df[metric], label=policy_name, marker='o')

    plt.xlabel('Step')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(f"Policy Comparison: {metric.replace('_', ' ').title()}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 1. Run the simulation to generate a log file
    #    python main.py --demo

    # 2. Load the produced simulation log
    df = pd.read_csv('outputs/simulation_log.csv')

    # 3. Plot collisions/alerts + fuel usage
    plot_simulation_stats(df, title='FuelSafe Demo Metrics')

    # Optional: compare baseline/rule_based if both are available
    # baseline_df = pd.read_csv('outputs/baseline_simulation_log.csv')
    # rule_based_df = pd.read_csv('outputs/rule_based_simulation_log.csv')
    # compare_policies({'baseline': baseline_df, 'rule_based': rule_based_df}, metric='fuel_used')
