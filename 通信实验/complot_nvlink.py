import matplotlib.pyplot as plt
import numpy as np

# 1. Data Preparation
# Sorted by SM count: 10, 15, 20, 25, 30
sm_counts = np.array([10, 15, 20, 25, 30])
total_sm = 108  # Updated for A100

# Baseline Data (from your input)
baseline_raw = np.array([381.43, 381.69, 383.96, 379.40, 383.61])
avg_baseline = np.mean(baseline_raw)
baseline_latency = np.full_like(sm_counts, avg_baseline, dtype=float)

# Measured Data (Interference)
interference_latency = np.array([416.49, 458.28, 507.69, 571.92, 699.43])

# Theoretical Calculation
# Formula: Time_theo = Time_base * (Total / (Total - Removed))
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

# 4. Fill Between (Visual only)
plt.fill_between(sm_counts, theoretical_latency, interference_latency,
                 color='orange', alpha=0.15, hatch='//', edgecolor='orange')

# 5. Annotations (Dual Labeling)
# Adjusted offsets for the new scale (approx 300ms - 700ms range)

for i in range(len(sm_counts)):
    # 5.1 Measured Annotation (Above the red point)
    # Offset adjusted from 150 to 15 because the data scale is much smaller now
    plt.text(sm_counts[i], interference_latency[i] + 15,
             f"+{actual_increase_pct[i]:.1f}%",
             ha='center', va='bottom', fontsize=9,
             color=color_measure, fontweight='bold')

    # 5.2 Theoretical Annotation (Below the blue point)
    # Offset adjusted from 300 to 25
    plt.text(sm_counts[i], theoretical_latency[i] - 25,
             f"+{theo_increase_pct[i]:.1f}%",
             ha='center', va='top', fontsize=9,
             color=color_theory, fontweight='bold')

# 6. Axes and Layout
plt.xlabel('Number of Dedicated Communication SMs (Total SMs: 108)', fontsize=12)
plt.ylabel('GEMM Computation Time (ms)', fontsize=12)
plt.title('Impact of SM Binding for Communication on Compute Performance (NVLink)', fontsize=13, weight='bold', pad=15)

plt.xticks(sm_counts)

# Adjusted Y-axis limit for the new data range (300ms to ~800ms)
plt.ylim(300, 850)
plt.grid(True, linestyle=':', alpha=0.6)

# Legend
plt.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=1, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig('nvlink.png', dpi=300, bbox_inches='tight')
plt.show()