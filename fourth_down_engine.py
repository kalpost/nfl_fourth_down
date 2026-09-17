import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class FourthDownAnalyzer:
    def expected_points(self, yardline):
        """
        Calibrated empirical NFL Expected Points curve.
        Yardline: 1 (own 1-yd line) to 99 (opponent 1-yd line).
        """
        y = np.clip(yardline, 1, 99)
        return -1.35 + 0.048 * y + 0.00025 * (y ** 2)

    def fg_success_prob(self, yardline):
        """
        Calibrated to modern NFL kicking success rates:
        - 33-yd FG (opp 16): ~98%
        - 42-yd FG (opp 25): ~89%
        - 50-yd FG (opp 33): ~78%
        - 60-yd FG (opp 43): ~48%
        """
        kick_distance = (100 - yardline) + 17
        return 1.0 / (1.0 + np.exp(0.14 * (kick_distance - 59.5)))

    def conversion_prob(self, yards_to_go):
        """
        Calibrated NFL 4th-down conversion rates:
        - 4th & 1: ~68%
        - 4th & 2: ~58%
        - 4th & 5: ~42%
        - 4th & 9: ~26%
        """
        k = max(1, yards_to_go)
        return float(np.clip(0.72 * np.exp(-0.115 * k), 0.08, 0.75))

    def evaluate_decisions(self, yardline, yards_to_go):
        # 1. Option: Go For It
        p_conv = self.conversion_prob(yards_to_go)
        fresh_set_bonus = 0.65 if yardline >= 60 else 0.25
        
        ep_success = self.expected_points(min(99, yardline + yards_to_go)) + fresh_set_bonus
        ep_failure = -self.expected_points(100 - yardline)
        ev_go = (p_conv * ep_success) + ((1.0 - p_conv) * ep_failure)

        # 2. Option: Field Goal
        p_fg = self.fg_success_prob(yardline)
        ep_fg_made = 3.0 - self.expected_points(30)
        kick_spot = max(20, yardline - 7)
        ep_fg_miss = -self.expected_points(100 - kick_spot)
        ev_fg = (p_fg * ep_fg_made) + ((1.0 - p_fg) * ep_fg_miss)

        # 3. Option: Punt
        if yardline >= 60:
            ev_punt = -999.0
        else:
            landed = min(80, yardline + 38)
            ev_punt = -self.expected_points(100 - landed)

        options = {"Go For It": ev_go, "Field Goal": ev_fg, "Punt": ev_punt}
        best_decision = max(options, key=options.get)

        return {
            "Optimal_Decision": best_decision,
            "EV_Go": round(ev_go, 3),
            "EV_FG": round(ev_fg, 3),
            "EV_Punt": round(ev_punt, 3),
            "Conv_Prob": round(p_conv, 3),
            "FG_Prob": round(p_fg, 3)
        }

    def generate_policy_matrix(self, max_yards=10, filename="decision_frontier.png"):
        """
        Evaluates decisions over a 2D coordinate grid:
        Yardlines: 20 (own 20) to 95 (opponent 5)
        Yards to Go: 1 to 10
        """
        yardlines = np.arange(20, 95)
        to_go = np.arange(1, max_yards + 1)
        matrix = np.zeros((len(to_go), len(yardlines)))

        decision_map = {"Punt": 0, "Field Goal": 1, "Go For It": 2}

        for i, k in enumerate(to_go):
            for j, y in enumerate(yardlines):
                res = self.evaluate_decisions(y, k)
                matrix[i, j] = decision_map[res["Optimal_Decision"]]

        # Visualization setup
        plt.figure(figsize=(11, 6), dpi=300)
        cmap = ListedColormap(["#2b5c8f", "#d95f02", "#1b9e77"])  # Blue (Punt), Orange (FG), Green (Go)
        
        plt.imshow(matrix, cmap=cmap, aspect="auto", origin="lower",
                   extent=[20, 95, 1, max_yards])
        
        cbar = plt.colorbar(ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(["Punt (0)", "Field Goal (1)", "Go For It (2)"])
        
        plt.title("NFL 4th Down Optimal Policy Frontier (Calibrated EV Model)", fontsize=13, fontweight="bold")
        plt.xlabel("Field Position (Yards from Own End Zone | 50 = Midfield)", fontsize=11)
        plt.ylabel("Yards to Go (Distance)", fontsize=11)
        plt.yticks(to_go)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        print(f"Heatmap successfully generated and saved as '{filename}'")

if __name__ == "__main__":
    analyzer = FourthDownAnalyzer()

    # Scenario 1: 4th & 1 at Opponent 25 (yardline = 75)
    print("--- Scenario 1: 4th & 1 at Opponent 25 ---")
    s1 = analyzer.evaluate_decisions(yardline=75, yards_to_go=1)
    for k, v in s1.items():
        print(f"{k}: {v}")

    # Scenario 2: 4th & 9 at Opponent 25 (yardline = 75)
    print("\n--- Scenario 2: 4th & 9 at Opponent 25 ---")
    s2 = analyzer.evaluate_decisions(yardline=75, yards_to_go=9)
    for k, v in s2.items():
        print(f"{k}: {v}")

    # Generate the actual heatmap image for your repo
    print("\nGenerating decision boundary matrix...")
    analyzer.generate_policy_matrix()