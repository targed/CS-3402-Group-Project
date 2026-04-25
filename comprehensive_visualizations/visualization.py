import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set a professional style
sns.set_theme(style="whitegrid")

# ==========================================
# CHART 1: The Language Gap (Zero-Shot)
# For Slide: "The Language Gap (Zero-Shot)"
# ==========================================
models_zs = ['Gemma-270m', 'Qwen-14B', 'Gemini-3-Flash']
en_f1_zs = [9.96, 55.48, 75.54]
es_f1_zs =[6.88, 33.40, 70.21]

x = np.arange(len(models_zs))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x - width/2, en_f1_zs, width, label='English', color='#1f77b4')
rects2 = ax.bar(x + width/2, es_f1_zs, width, label='Spanish', color='#d62728')

ax.set_ylabel('F1 Score')
ax.set_title('Zero-Shot Performance: The "Language Tax"', fontsize=14, pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models_zs, fontsize=12)
ax.set_ylim(0, 100)
ax.legend()

# Add text labels on top of bars
ax.bar_label(rects1, padding=3, fmt='%.1f')
ax.bar_label(rects2, padding=3, fmt='%.1f')

plt.tight_layout()
plt.savefig('zero_shot_gap.png', dpi=300)
plt.show()

# ==========================================
# CHART 2: The Fine-Tuning Solution
# For Slide: "The Fine-Tuning Solution"
# ==========================================
# Comparing Qwen 0.5B Zero-Shot vs Fine-Tuned
stages =['Base (Zero-Shot)', 'Fine-Tuned (LoRA)']
en_f1_ft =[19.23, 72.23]
es_f1_ft =[5.12, 65.40]

x2 = np.arange(len(stages))

fig2, ax2 = plt.subplots(figsize=(8, 5))
rects3 = ax2.bar(x2 - width/2, en_f1_ft, width, label='English', color='#1f77b4')
rects4 = ax2.bar(x2 + width/2, es_f1_ft, width, label='Spanish', color='#d62728')

ax2.set_ylabel('F1 Score')
ax2.set_title('Impact of Fine-Tuning on Qwen1.5-0.5B', fontsize=14, pad=15)
ax2.set_xticks(x2)
ax2.set_xticklabels(stages, fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend()

# Add text labels
ax2.bar_label(rects3, padding=3, fmt='%.1f')
ax2.bar_label(rects4, padding=3, fmt='%.1f')

# Add an annotation showing the gap shrank
ax2.annotate('Gap: 14.1 pts', xy=(0, 25), xytext=(0, 35),
            arrowprops=dict(facecolor='black', shrink=0.05), ha='center')
ax2.annotate('Gap: 6.8 pts', xy=(1, 75), xytext=(1, 85),
            arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

plt.tight_layout()
plt.savefig('fine_tuning_impact.png', dpi=300)
plt.show()