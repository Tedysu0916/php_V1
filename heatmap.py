import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generate_horizontal_heatmap(word_expert_usage, num_experts, save_path="heatmap_horizontal.png"):
    """横向布局的渐变分组热力图 - 适合论文"""

    semantic_groups = {
        'Color': (['black', 'red', 'blue'], '#FFCDD2', '#E53935'),
        'Clothing': (['jeans', 'dress', 'jacket', 'coat'], '#BBDEFB', '#1E88E5'),
        'Accessory': (['sneakers', 'backpack'], '#C8E6C9', '#43A047'),
        'Person': (['man', 'woman', 'person'], '#FFE0B2', '#FB8C00'),
        'Age': (['young', 'elderly'], '#E1BEE7', '#8E24AA'),
        'Action': (['wearing'], '#FFF9C4', '#F9A825'),
    }

    sorted_words = []
    group_info = []

    for group_name, (words, light_color, dark_color) in semantic_groups.items():
        start = len(sorted_words)
        for word in words:
            if word in word_expert_usage:
                sorted_words.append(word)
        if len(sorted_words) > start:
            group_info.append((start, len(sorted_words), group_name, light_color, dark_color))

    if not sorted_words:
        print("⚠️ 没有找到任何关键词！")
        return

    # 转置数据：行=专家，列=关键词
    heatmap_data = np.array([word_expert_usage[word] for word in sorted_words]).T
    max_val = heatmap_data.max()

    # 横向布局：宽 > 高
    fig, ax = plt.subplots(figsize=(16, 5))

    # 绘制分组背景（垂直方向）
    for start, end, name, light_color, dark_color in group_info:
        ax.axvspan(start - 0.5, end - 0.5, facecolor=light_color, alpha=0.4, zorder=0)

    # 绘制热力图单元格
    for j in range(len(sorted_words)):  # 列=关键词
        group_color = None
        for start, end, name, light_color, dark_color in group_info:
            if start <= j < end:
                group_color = dark_color
                break

        for i in range(num_experts):  # 行=专家
            value = heatmap_data[i, j]
            intensity = value / max_val if max_val > 0 else 0

            rect = mpatches.FancyBboxPatch(
                (j - 0.4, i - 0.4), 0.8, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor=group_color if value > 0 else '#E0E0E0',
                alpha=0.3 + intensity * 0.7 if value > 0 else 0.3,
                edgecolor='white',
                linewidth=2,
                zorder=1
            )
            ax.add_patch(rect)

            text_color = 'white' if intensity > 0.5 else ('black' if value > 0 else 'gray')
            fontsize = 12 if value > 0 else 10
            ax.text(j, i, f'{int(value)}', ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=text_color, zorder=2)

    ax.set_xlim(-0.5, len(sorted_words) - 0.5)
    ax.set_ylim(-0.5, num_experts - 0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    # X轴=关键词，Y轴=专家
    ax.set_xticks(range(len(sorted_words)))
    ax.set_xticklabels(sorted_words, fontsize=11, rotation=45, ha='right')
    ax.set_yticks(range(num_experts))
    ax.set_yticklabels([f'Expert {i}' for i in range(num_experts)], fontsize=12, fontweight='bold')

    # 顶部添加分组标签
    for start, end, name, light_color, dark_color in group_info:
        mid = (start + end - 1) / 2
        ax.text(mid, -0.9, name, va='bottom', ha='center',
                fontsize=11, fontweight='bold', color=dark_color,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=light_color,
                          edgecolor=dark_color, alpha=0.9, linewidth=1.5))

    ax.set_title('Expert Specialization by Semantic Categories',
                 fontsize=16, fontweight='bold', pad=35)
    ax.set_ylabel('Experts', fontsize=13, fontweight='bold')
    ax.set_xlabel('Keywords', fontsize=13, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


def generate_horizontal_heatmap_v2(word_expert_usage, num_experts, save_path="heatmap_horizontal_v2.png"):
    """横向布局方案2 - 更紧凑，分组线条"""

    semantic_groups = {
        'Color': (['black', 'red', 'blue'], '#E53935'),
        'Clothing': (['jeans', 'dress', 'jacket', 'coat'], '#1E88E5'),
        'Accessory': (['sneakers', 'backpack'], '#43A047'),
        'Person': (['man', 'woman', 'person'], '#FB8C00'),
        'Age': (['young', 'elderly'], '#8E24AA'),
        'Action': (['wearing'], '#F9A825'),
    }

    sorted_words = []
    group_info = []
    word_colors = {}

    for group_name, (words, color) in semantic_groups.items():
        start = len(sorted_words)
        for word in words:
            if word in word_expert_usage:
                sorted_words.append(word)
                word_colors[word] = color
        if len(sorted_words) > start:
            group_info.append((start, len(sorted_words), group_name, color))

    if not sorted_words:
        print("⚠️ 没有找到任何关键词！")
        return

    heatmap_data = np.array([word_expert_usage[word] for word in sorted_words]).T
    max_val = heatmap_data.max()

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # 绘制单元格
    for j in range(len(sorted_words)):
        color = word_colors[sorted_words[j]]
        for i in range(num_experts):
            value = heatmap_data[i, j]
            intensity = value / max_val if max_val > 0 else 0

            rect = mpatches.FancyBboxPatch(
                (j - 0.42, i - 0.42), 0.84, 0.84,
                boxstyle="round,pad=0.01,rounding_size=0.12",
                facecolor=color if value > 0 else '#EEEEEE',
                alpha=0.25 + intensity * 0.75 if value > 0 else 0.4,
                edgecolor='white',
                linewidth=1.5,
                zorder=1
            )
            ax.add_patch(rect)

            text_color = 'white' if intensity > 0.45 else ('black' if value > 0 else '#999')
            ax.text(j, i, f'{int(value)}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color, zorder=2)

    # 添加分组分隔线
    for start, end, name, color in group_info:
        if start > 0:
            ax.axvline(x=start - 0.5, color='#666666', linewidth=1.5, linestyle='--', alpha=0.6)

    ax.set_xlim(-0.5, len(sorted_words) - 0.5)
    ax.set_ylim(-0.5, num_experts - 0.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    ax.set_xticks(range(len(sorted_words)))
    ax.set_xticklabels(sorted_words, fontsize=10, rotation=40, ha='right')
    ax.set_yticks(range(num_experts))
    ax.set_yticklabels([f'Expert {i}' for i in range(num_experts)], fontsize=11, fontweight='bold')

    # 顶部分组标签
    for start, end, name, color in group_info:
        mid = (start + end - 1) / 2
        ax.text(mid, -0.75, name, va='bottom', ha='center',
                fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.25', facecolor=color,
                          edgecolor='none', alpha=0.9))

    # ax.set_title('Expert Specialization by Semantic Categories',
    #              fontsize=14, fontweight='bold', pad=30)
    # ax.set_ylabel('Experts', fontsize=12, fontweight='bold')

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved: {save_path}")


if __name__ == "__main__":
    generate_horizontal_heatmap(word_expert_usage, num_experts, "heatmap_horizontal.png")
    generate_horizontal_heatmap_v2(word_expert_usage, num_experts, "heatmap_horizontal_v2.png")
    print("\n�� 横向热力图已生成！")