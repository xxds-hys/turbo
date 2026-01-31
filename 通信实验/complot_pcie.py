import matplotlib.pyplot as plt
import numpy as np

# 1. Data Preparation
sm_counts = np.array([10, 15, 20, 25, 30])
total_sm = 82

# Baseline (Average)
baseline_raw = np.array([5997.45, 5993.42, 6010.96, 5997.20, 5995.42])
avg_baseline = np.mean(baseline_raw)
baseline_latency = np.full_like(sm_counts, avg_baseline, dtype=float)

# Measured Data (Interference)
interference_latency = np.array([6811.05, 7404.77, 8125.54, 9172.89, 10331.18])

# Theoretical Calculation
theoretical_latency = avg_baseline * (total_sm / (total_sm - sm_counts))

# Calculate Percentages
actual_increase_pct = (interference_latency - avg_baseline) / avg_baseline * 100
theo_increase_pct = (theoretical_latency - avg_baseline) / avg_baseline * 100

# 2. Setup Plot Style (Academic)
plt.figure(figsize=(9, 6), dpi=300)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# Colors
color_measure = '#D62728'  # Red
color_theory = '#1F77B4'   # Blue
color_base = '#2CA02C'     # Green

# 3. Plotting Lines

# A. Baseline
plt.plot(sm_counts, baseline_latency, linestyle='--', color=color_base, linewidth=1.5,
         label='Baseline (No Comm)')

# B. Theoretical
plt.plot(sm_counts, theoretical_latency, marker='^', markersize=7, linestyle='-.',
         color=color_theory, linewidth=1.5, label='Theoretical (Caused by SM Reduction)')

# C. Measured
plt.plot(sm_counts, interference_latency, marker='s', markersize=7, linestyle='-',
         color=color_measure, linewidth=2, label='Measured (With Comm)')

# 4. Fill Between (Visual only, no legend)
# Note: label is omitted so it doesn't appear in the legend
plt.fill_between(sm_counts, theoretical_latency, interference_latency,
                 color='orange', alpha=0.15, hatch='//', edgecolor='orange')

# 5. Annotations (Dual Labeling)

for i in range(len(sm_counts)):
    # 5.1 Measured Annotation (Above the red point)
    # Using bold font to emphasize the experimental result
    plt.text(sm_counts[i], interference_latency[i] + 150,
             f"+{actual_increase_pct[i]:.1f}%",
             ha='center', va='bottom', fontsize=9,
             color=color_measure, fontweight='bold')

    # 5.2 Theoretical Annotation (Below the blue point)
    # Using a slightly smaller font and "Theory" prefix for clarity
    # Added negative offset (-300) to push text down
    plt.text(sm_counts[i], theoretical_latency[i] - 300,
             f"+{theo_increase_pct[i]:.1f}%",
             ha='center', va='top', fontsize=9,
             color=color_theory, fontweight='bold')

# 6. Axes and Layout
plt.xlabel('Number of Dedicated Communication SMs(Total SMs: 82)', fontsize=12)
plt.ylabel('GEMM Computation Time (ms)', fontsize=12)
plt.title('The Impact of SM Binding for Communication on Compute Performance(PCIe )', fontsize=13, weight='bold', pad=15)

plt.xticks(sm_counts)
# Y-axis limit adjusted slightly to make room for bottom labels if needed
plt.ylim(5000, 11500)
plt.grid(True, linestyle=':', alpha=0.6)

# Legend (Clean, only 3 entries)
plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('pcie.png', dpi=300, bbox_inches='tight')
plt.show()