import os
import numpy as np
import matplotlib.pyplot as plt

# ================== 全局风格 ==================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 配色方案
COLOR_BEFORE = '#D62728'  # 红色
COLOR_AFTER = '#1f77b4'   # 蓝色

SAVE_DIR = './moe_radar_efp'
os.makedirs(SAVE_DIR, exist_ok=True)


# ================== 雷达图函数 ==================
def plot_moe_radar(experts, before, after, title, sub_label, save_path):
    labels = experts
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, angles[:1]])

    before_closed = before + before[:1]
    after_closed = after + after[:1]

    fig = plt.figure(figsize=(6.5, 7.0))
    ax = plt.subplot(111, polar=True)

    # 数据绘制部分保持不变...
    ax.plot(angles, before_closed, linewidth=3, marker='o', markersize=7,
            label='Without $\mathcal{L}_{aux}$', color=COLOR_BEFORE, zorder=2)
    ax.fill(angles, before_closed, color=COLOR_BEFORE, alpha=0.15)
    ax.plot(angles, after_closed, linewidth=3, marker='s', markersize=7,
            label='With $\mathcal{L}_{aux}$', color=COLOR_AFTER, zorder=3)
    ax.fill(angles, after_closed, color=COLOR_AFTER, alpha=0.25)

    # -------- 重点修改区域 --------
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=18)
    ax.set_ylim(0, 0.6)
    ax.set_yticks([0.2, 0.4]) # 控制网格线位置
    ax.set_yticklabels([])    # 移除刻度显示
    ax.grid(True, linestyle='--', alpha=0.4) # 弱化网格线
    # ----------------------------

    ax.set_title(title, fontsize=20, pad=35, fontweight='bold')
    plt.figtext(0.5, 0.02, sub_label, wrap=True, horizontalalignment='center', fontsize=22)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.15), ncol=1, fontsize=14, frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ================== 数据准备 ==================
experts = ['Expert 0', 'Expert 1', 'Expert 2', 'Expert 3']

# Without L_{aux}
text_before = {
    0: [0.1051, 0.1224, 0.2734, 0.4991],
    1: [0.3933, 0.2332, 0.2152, 0.1583],
    2: [0.0550, 0.4022, 0.0429, 0.5000],
    3: [0.4550, 0.1159, 0.1075, 0.3216],
}
image_before = {
    0: [0.1353, 0.1196, 0.2458, 0.4993],
    1: [0.3972, 0.2398, 0.1977, 0.1654],
    2: [0.0494, 0.3952, 0.0554, 0.5000],
    3: [0.4625, 0.1126, 0.1148, 0.3101],
}

# With L_{aux}
text_after = {
    0: [0.2492, 0.2608, 0.3197, 0.1703],
    1: [0.2249, 0.2441, 0.2856, 0.2454],
    2: [0.2805, 0.2509, 0.2731, 0.1955],
    3: [0.2544, 0.2315, 0.2460, 0.2681],
}
image_after = {
    0: [0.2525, 0.2563, 0.2995, 0.1917],
    1: [0.2440, 0.2481, 0.2775, 0.2305],
    2: [0.2786, 0.2486, 0.2801, 0.1927],
    3: [0.2882, 0.2126, 0.2355, 0.2638],
}

# ================== 绘图 ==================
# 按照你给的样例图顺序：第一排 a-d 是 Image，第二排 e-h 是 Text
sub_labels_img = ['(a)', '(b)', '(c)', '(d)']
sub_labels_txt = ['(e)', '(f)', '(g)', '(h)']

for layer in range(4):
    # Image EFP Module
    plot_moe_radar(
        experts,
        image_before[layer],
        image_after[layer],
        f'Image EFP Module {layer}',
        sub_labels_img[layer],
        os.path.join(SAVE_DIR, f'image_mod{layer}.png')
    )

    # Text EFP Module
    plot_moe_radar(
        experts,
        text_before[layer],
        text_after[layer],
        f'Text EFP Module {layer}',
        sub_labels_txt[layer],
        os.path.join(SAVE_DIR, f'text_mod{layer}.png')
    )

print(f"所有 8 张图已生成，标题和子图编号已添加。路径: {SAVE_DIR}")