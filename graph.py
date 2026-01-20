import matplotlib.pyplot as plt

# 維持學術風格參數
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'figure.autolayout': True
})

# 數據
steps = [20, 40, 60, 80, 100]
val_loss = [0.19988, 0.07965, 0.03123, 0.06927, 0.03880]
train_loss = [0.1955, 0.0460, 0.0828, 0.0096, 0.0108]
test_score = [0.8265, 0.9637, 0.9780, 0.9778, 0.9778]

# 建立圖表
fig, ax1 = plt.subplots(figsize=(8, 6))

# --- 1. 調整「收斂穩定區」背景 (Convergence Zone) ---
# [修改] 範圍從 70 改為 60 開始，涵蓋 60-105
ax1.axvspan(60, 105, color='gray', alpha=0.15, lw=0)

# [修改] 文字位置水平置中調整 (60+105)/2 = 82.5
ax1.text(82.5, 0.26, "Convergence Phase", ha='center', va='center', 
         fontsize=11, style='italic', color='dimgray', fontweight='bold')

# --- 左軸: Loss ---
color_train = '#1f77b4'
color_val = '#000080'
ax1.set_xlabel('Training Steps', fontweight='bold')
ax1.set_ylabel('Loss', color='black', fontweight='bold')

l1, = ax1.plot(steps, train_loss, color=color_train, linestyle='--', marker='o', label='Training Loss')
l2, = ax1.plot(steps, val_loss, color=color_val, linestyle='-', marker='s', label='Validation Loss')

ax1.tick_params(axis='y', labelcolor='black')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.set_ylim(0, 0.28) # 保持較高上限，避開文字

# --- 右軸: Score ---
ax2 = ax1.twinx()
color_score = '#d62728'
ax2.set_ylabel('DWA Score', color=color_score, fontweight='bold')

l3, = ax2.plot(steps, test_score, color=color_score, linestyle='-', marker='^', label='Test DWA Score')

ax2.tick_params(axis='y', labelcolor=color_score)
ax2.set_ylim(0.80, 1.02) # 保持適合的範圍

# --- 2. 效能高原標註 ---
# 維持指向最後一點
ax2.annotate('Stable Peak Performance\n(DWA $\\approx$ 0.978)', 
             xy=(99.5, 0.9766), xycoords='data',
             xytext=(70, 0.95), textcoords='data',
             arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
             fontsize=10, color='black', fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

# --- X 軸設定 ---
ax1.set_xlim(15, 105)

# --- 圖例 ---
lines = [l1, l2, l3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
           ncol=3, frameon=False)

plt.savefig('academic_plot_stability_v3_step60.png', dpi=300)
plt.show()