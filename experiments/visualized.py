"""
攻击效果可视化模块
"""
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import seaborn as sns


class AttackVisualizer:
    """攻击效果可视化工具"""

    def __init__(self, result):
        self.result = result

    def plot_rank_changes(self, save_path: Optional[str] = None):
        """绘制目标项的排名变化"""
        fig, ax = plt.subplots(figsize=(10, 6))

        targets = self.result.target_items
        before_ranks = [self.result.before_ranks.get(t, 0) for t in targets]
        after_ranks = [self.result.after_ranks.get(t, 0) for t in targets]

        x = np.arange(len(targets))
        width = 0.35

        bars1 = ax.bar(x - width / 2, before_ranks, width, label='Before Attack', color='skyblue')
        bars2 = ax.bar(x + width / 2, after_ranks, width, label='After Attack', color='salmon')

        ax.set_xlabel('Target Items')
        ax.set_ylabel('Rank (lower is better)')
        ax.set_title(f'Rank Changes - {self.result.protocol.upper()} ({self.result.attack} attack)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Item {t}' for t in targets])
        ax.legend()

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_frequency_heatmap(self, top_k: int = 20, save_path: Optional[str] = None):
        """绘制频率变化热力图"""
        # 获取Top-K项目的频率变化
        all_items = set(self.result.before_freq.keys()) | set(self.result.after_freq.keys())

        changes = []
        for item in all_items:
            before = self.result.before_freq.get(item, 0)
            after = self.result.after_freq.get(item, 0)
            changes.append((item, after - before, before, after))

        # 按变化幅度排序
        changes.sort(key=lambda x: abs(x[1]), reverse=True)
        top_changes = changes[:top_k]

        items = [f'Item {c[0]}' for c in top_changes]
        before_vals = [c[2] for c in top_changes]
        after_vals = [c[3] for c in top_changes]

        fig, ax = plt.subplots(figsize=(12, 8))

        # 创建热力图数据
        data = np.array([before_vals, after_vals])

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0)

        ax.set_xticks(np.arange(len(items)))
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['Before', 'After'])
        ax.set_xticklabels(items, rotation=45, ha='right')
        ax.set_title(f'Frequency Changes - {self.result.protocol.upper()}')

        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Frequency')

        # 添加数值标签
        for i in range(len(items)):
            for j in range(2):
                text = ax.text(i, j, int(data[j, i]),
                               ha="center", va="center", color="black")

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def generate_report(self, output_dir: str = "attack_reports"):
        """生成完整的攻击报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存可视化
        self.plot_rank_changes(save_path=f"{output_dir}/rank_changes.png")
        self.plot_frequency_heatmap(save_path=f"{output_dir}/frequency_heatmap.png")

        # 生成文本报告
        report_path = f"{output_dir}/attack_report.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("  LDP Poison Attack Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(self.result.summary())
            f.write("\n\n")
            f.write("-" * 40 + "\n")
            f.write("Detailed Target Item Analysis:\n")
            f.write("-" * 40 + "\n")

            for t in self.result.target_items:
                before_f = self.result.before_freq.get(t, 0)
                after_f = self.result.after_freq.get(t, 0)
                before_r = self.result.before_ranks.get(t, '?')
                after_r = self.result.after_ranks.get(t, '?')
                f.write(f"Item {t:3d}: freq {before_f:4d} → {after_f:4d} (+{after_f - before_f:+d}) | "
                        f"rank {before_r:2d} → {after_r:2d} ({before_r - after_r:+d})\n")

        return report_path