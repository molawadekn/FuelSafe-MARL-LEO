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


def compute_metrics(df):
    """Compute performance metrics from simulation log."""
    total_steps = len(df)
    total_collisions = df['collisions'].iloc[-1] if len(df) > 0 else 0
    total_fuel = df['fuel_used'].iloc[-1] if len(df) > 0 else 0
    total_alerts = df['alerts'].iloc[-1] if len(df) > 0 else 0
    
    # Success rate: proportion of steps without collisions
    success_rate = (total_steps - total_collisions) / total_steps if total_steps > 0 else 0
    
    # Efficiency: collisions avoided per fuel unit
    max_possible_collisions = total_steps  # assume every step could have collision
    avoided = max_possible_collisions - total_collisions
    efficiency = avoided / (total_fuel + 1e-6)  # avoid division by zero
    
    # Accuracy: same as success rate for binary classification
    accuracy = success_rate
    
    # F1 score: treat alerts as predictions of collisions
    # True positive: steps with alert and collision
    # False positive: steps with alert but no collision
    # False negative: steps with collision but no alert
    # True negative: steps with no alert and no collision
    
    alerts_series = df['alerts'] > 0  # binary alert presence
    collisions_series = df['collisions'] > 0  # binary collision presence
    
    tp = ((alerts_series) & (collisions_series)).sum()
    fp = ((alerts_series) & (~collisions_series)).sum()
    fn = ((~alerts_series) & (collisions_series)).sum()
    tn = ((~alerts_series) & (~collisions_series)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_collisions': total_collisions,
        'total_fuel': total_fuel,
        'total_alerts': total_alerts,
        'success_rate': success_rate,
        'efficiency': efficiency,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


def plot_metrics_comparison(policy_metrics):
    """Plot bar charts comparing metrics across policies."""
    metrics = ['total_collisions', 'total_fuel', 'success_rate', 'efficiency', 'accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Total Collisions', 'Total Fuel (kg)', 'Success Rate', 'Efficiency', 'Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    policy_names = list(policy_metrics.keys())
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [policy_metrics[p][metric] for p in policy_names]
        axes[i].bar(policy_names, values, color=['skyblue', 'lightcoral', 'lightgreen'][:len(policy_names)])
        axes[i].set_title(label)
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)
    
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
    # 1. Run the simulation to generate log files
    #    python main.py --demo

    # 2. Load the produced simulation logs
    baseline_df = pd.read_csv('outputs/baseline_simulation_log.csv')
    rule_based_df = pd.read_csv('outputs/rule_based_simulation_log.csv')

    # 3. Plot individual stats
    plot_simulation_stats(baseline_df, title='Baseline Policy Metrics')
    plot_simulation_stats(rule_based_df, title='Rule-Based Policy Metrics')

    # 4. Compute and plot metrics comparison
    policy_metrics = {
        'baseline': compute_metrics(baseline_df),
        'rule_based': compute_metrics(rule_based_df)
    }
    
    plot_metrics_comparison(policy_metrics)

    # 5. Optional: compare time-series for specific metrics
    compare_policies({'baseline': baseline_df, 'rule_based': rule_based_df}, metric='fuel_used')
    compare_policies({'baseline': baseline_df, 'rule_based': rule_based_df}, metric='collisions')
