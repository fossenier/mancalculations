"""
Real-time monitoring and visualization for AlphaZero Kalah training
Provides web interface and command-line tools for tracking progress
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
from typing import Dict, List, Optional
import pandas as pd
from flask import Flask, render_template, jsonify
import threading
import time


class TrainingMonitor:
    """Monitor training progress and statistics"""

    def __init__(self, log_dir: str, checkpoint_dir: str):
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.stats_cache = {}

    def get_training_stats(self) -> Dict[str, any]:
        """Gather all training statistics"""
        stats = {
            "iterations": [],
            "evaluations": [],
            "checkpoints": [],
            "current_iteration": 0,
            "training_time": 0,
            "best_model": None,
        }

        # Read evaluation files
        eval_files = [
            f for f in os.listdir(self.log_dir) if f.startswith("evaluation_")
        ]
        for eval_file in eval_files:
            with open(os.path.join(self.log_dir, eval_file), "r") as f:
                eval_data = json.load(f)
                stats["evaluations"].append(eval_data)

        # Read training state
        state_file = os.path.join(self.checkpoint_dir, "training_state.json")
        if os.path.exists(state_file):
            with open(state_file, "r") as f:
                state = json.load(f)
                stats["current_iteration"] = state["iteration"]
                stats["best_model"] = state.get("best_model_path")

        # List checkpoints
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")]
        stats["checkpoints"] = sorted(checkpoints)

        return stats

    def plot_training_curves(self, save_path: Optional[str] = None):
        """Generate training curve plots"""
        stats = self.get_training_stats()

        if not stats["evaluations"]:
            print("No evaluation data found")
            return

        # Extract data for plotting
        iterations = []
        win_rates = {
            "Random": [],
            "Minimax(depth=3)": [],
            "Minimax(depth=5)": [],
            "Minimax(depth=7)": [],
        }

        for eval_data in sorted(stats["evaluations"], key=lambda x: x["iteration"]):
            iterations.append(eval_data["iteration"])

            for opponent, results in eval_data["matches"].items():
                if opponent in win_rates:
                    win_rates[opponent].append(results["win_rate"])

        # Create plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Win rate plot
        for opponent, rates in win_rates.items():
            if rates:
                ax1.plot(iterations[: len(rates)], rates, marker="o", label=opponent)

        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Win Rate")
        ax1.set_title("Win Rate vs Different Opponents")
        ax1.legend()
        ax1.grid(True)
        ax1.set_ylim([0, 1])

        # Score difference plot
        score_diffs = []
        for eval_data in sorted(stats["evaluations"], key=lambda x: x["iteration"]):
            avg_diff = np.mean(
                [
                    results.get("avg_score_diff", 0)
                    for results in eval_data["matches"].values()
                ]
            )
            score_diffs.append(avg_diff)

        ax2.plot(iterations[: len(score_diffs)], score_diffs, marker="s", color="green")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Average Score Difference")
        ax2.set_title("Average Score Advantage")
        ax2.grid(True)
        ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def generate_report(self, output_file: str):
        """Generate comprehensive training report"""
        stats = self.get_training_stats()

        report = f"""
# AlphaZero Kalah Training Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Progress
- Current Iteration: {stats['current_iteration']}
- Total Checkpoints: {len(stats['checkpoints'])}
- Best Model: {os.path.basename(stats['best_model'] or 'None')}

## Performance Summary
"""

        if stats["evaluations"]:
            latest_eval = max(stats["evaluations"], key=lambda x: x["iteration"])

            report += (
                f"\n### Latest Evaluation (Iteration {latest_eval['iteration']})\n\n"
            )
            report += "| Opponent | Win Rate | Avg Score | Score Diff |\n"
            report += "|----------|----------|-----------|------------|\n"

            for opponent, results in latest_eval["matches"].items():
                report += f"| {opponent} | {results['win_rate']:.2%} | "
                report += f"{results['avg_score']:.1f} | "
                report += f"{results.get('avg_score_diff', 0):.1f} |\n"

            if "mcts_scaling" in latest_eval:
                report += f"\n### MCTS Scaling\n"
                report += f"Win rate (1600 vs 400 sims): {latest_eval['mcts_scaling']['strong_vs_weak_win_rate']:.2%}\n"

        # Save report
        with open(output_file, "w") as f:
            f.write(report)

        print(f"Report saved to {output_file}")


# Web interface for real-time monitoring
app = Flask(__name__)
monitor = None


@app.route("/")
def index():
    """Main dashboard page"""
    return render_template("dashboard.html")


@app.route("/api/stats")
def get_stats():
    """API endpoint for current statistics"""
    return jsonify(monitor.get_training_stats())


@app.route("/api/latest_games")
def get_latest_games():
    """Get recent self-play games for visualization"""
    # Implementation would read recent game data
    return jsonify({"games": []})


def run_web_server(monitor_instance, port=5000):
    """Run web monitoring server"""
    global monitor
    monitor = monitor_instance
    app.run(host="0.0.0.0", port=port, debug=False)


# Dashboard HTML template
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>AlphaZero Kalah Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .stat-box { 
            border: 1px solid #ddd; 
            padding: 15px; 
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .stat-value { font-size: 24px; font-weight: bold; color: #333; }
        .stat-label { color: #666; }
        #performance-chart { width: 100%; height: 400px; margin-top: 20px; }
        .refresh-btn { 
            background-color: #4CAF50; 
            color: white; 
            border: none; 
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 4px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>AlphaZero Kalah Training Monitor</h1>
    
    <button class="refresh-btn" onclick="refreshStats()">Refresh</button>
    
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-label">Current Iteration</div>
            <div class="stat-value" id="current-iteration">-</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Total Checkpoints</div>
            <div class="stat-value" id="total-checkpoints">-</div>
        </div>
        <div class="stat-box">
            <div class="stat-label">Latest Win Rate vs Minimax-7</div>
            <div class="stat-value" id="latest-winrate">-</div>
        </div>
    </div>
    
    <canvas id="performance-chart"></canvas>
    
    <script>
        let chart = null;
        
        function refreshStats() {
            fetch('/api/stats')
                .then(response => response.json())
                .then(data => updateDashboard(data));
        }
        
        function updateDashboard(data) {
            document.getElementById('current-iteration').textContent = data.current_iteration;
            document.getElementById('total-checkpoints').textContent = data.checkpoints.length;
            
            // Update win rate
            if (data.evaluations.length > 0) {
                const latest = data.evaluations[data.evaluations.length - 1];
                const minimax7 = latest.matches['Minimax(depth=7)'];
                if (minimax7) {
                    document.getElementById('latest-winrate').textContent = 
                        (minimax7.win_rate * 100).toFixed(1) + '%';
                }
            }
            
            updateChart(data);
        }
        
        function updateChart(data) {
            // Prepare chart data
            // ... (chart implementation)
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshStats, 30000);
        refreshStats();
    </script>
</body>
</html>
"""


def main():
    """Command-line interface for monitoring"""
    parser = argparse.ArgumentParser(description="Monitor AlphaZero Kalah Training")
    parser.add_argument("--log-dir", default="./kalah_logs", help="Log directory")
    parser.add_argument(
        "--checkpoint-dir", default="./kalah_checkpoints", help="Checkpoint directory"
    )
    parser.add_argument("--plot", action="store_true", help="Generate training plots")
    parser.add_argument("--report", type=str, help="Generate report to file")
    parser.add_argument(
        "--web", action="store_true", help="Start web monitoring server"
    )
    parser.add_argument("--port", type=int, default=5000, help="Web server port")

    args = parser.parse_args()

    monitor = TrainingMonitor(args.log_dir, args.checkpoint_dir)

    if args.plot:
        monitor.plot_training_curves()
    elif args.report:
        monitor.generate_report(args.report)
    elif args.web:
        # Save dashboard template
        os.makedirs("templates", exist_ok=True)
        with open("templates/dashboard.html", "w") as f:
            f.write(dashboard_html)

        print(f"Starting web server on http://localhost:{args.port}")
        run_web_server(monitor, args.port)
    else:
        # Print current stats
        stats = monitor.get_training_stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
