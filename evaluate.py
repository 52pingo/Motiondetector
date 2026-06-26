#!/usr/bin/env python3
"""
运动目标检测评估脚本
--------------------
将模型输出的 Mask 与 Groundtruth 进行逐帧多指标对比，
生成统计图表和可视化结果。

评估指标：
  - IoU (Intersection over Union / Jaccard)
  - Precision, Recall, F1-Score
  - Accuracy
  - 逐帧 TP/FP/FN 分布

用法：
  python evaluate.py <pred_dir> <gt_dir> [--temporal-roi START END] [--init N]
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os, sys, argparse
from glob import glob
from collections import defaultdict

# ============================================================
# 0. 参数解析
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='运动目标检测评估')
    parser.add_argument('pred_dir', help='模型输出 Mask 目录')
    parser.add_argument('gt_dir', help='Groundtruth 目录')
    parser.add_argument('--temporal-roi', nargs=2, type=int, default=[300, 1200],
                        help='temporalROI 评估范围 (start end)')
    parser.add_argument('--init', type=int, default=50,
                        help='模型初始化帧数（背景建模消耗的帧数）')
    parser.add_argument('--output-dir', default='evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--prefix-pred', default='mask_',
                        help='预测 Mask 文件名前缀')
    parser.add_argument('--prefix-gt', default='gt',
                        help='Groundtruth 文件名前缀')
    parser.add_argument('--gt-ext', default='.png',
                        help='Groundtruth 文件扩展名')
    return parser.parse_args()


# ============================================================
# 1. 逐帧指标计算
# ============================================================

def compute_metrics(pred_mask, gt_mask):
    """
    计算单帧二分类指标。
    pred_mask / gt_mask: 二值图 (0=背景, 255=前景), 尺寸需一致。
    返回 dict: {tp, fp, fn, tn, precision, recall, f1, iou, accuracy}
    """
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin   = (gt_mask   > 127).astype(np.uint8)

    tp = np.sum((pred_bin == 1) & (gt_bin == 1))
    fp = np.sum((pred_bin == 1) & (gt_bin == 0))
    fn = np.sum((pred_bin == 0) & (gt_bin == 1))
    tn = np.sum((pred_bin == 0) & (gt_bin == 0))

    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)
    accuracy  = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall,
        'f1': f1, 'iou': iou, 'accuracy': accuracy
    }


def load_mask(path):
    """加载二值 Mask，转为灰度图 """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return img


# ============================================================
# 2. 主评估流程
# ============================================================

def run_evaluation(args):
    os.makedirs(args.output_dir, exist_ok=True)

    start_frame, end_frame = args.temporal_roi
    init = args.init

    print(f"评估范围: frame {start_frame} ~ {end_frame}")
    print(f"模型初始帧: {init}")
    print(f"预测目录: {args.pred_dir}")
    print(f"真值目录: {args.gt_dir}")

    metrics_all = []
    frame_nums = []

    for frame_id in range(start_frame, end_frame + 1):
        # 预测 Mask 文件名: mask_XXXXX.png (XXXXX = frame_id - init, 5位补零)
        pred_idx = frame_id - init
        if pred_idx < 0:
            continue
        pred_path = os.path.join(
            args.pred_dir,
            f"{args.prefix_pred}{pred_idx:05d}.png")

        # Groundtruth 文件名: gtXXXXXX.png
        gt_path = os.path.join(
            args.gt_dir,
            f"{args.prefix_gt}{frame_id:06d}{args.gt_ext}")

        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue

        pred = load_mask(pred_path)
        gt   = load_mask(gt_path)

        if pred is None or gt is None:
            continue

        # 尺寸对齐
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        metrics = compute_metrics(pred, gt)
        metrics['frame'] = frame_id
        metrics_all.append(metrics)
        frame_nums.append(frame_id)

    if not metrics_all:
        print("错误：未找到可匹配的帧对，请检查路径和参数。")
        sys.exit(1)

    print(f"成功评估 {len(metrics_all)} 帧")

    # 汇总指标
    keys = ['precision', 'recall', 'f1', 'iou', 'accuracy']
    summary = {}
    for k in keys:
        vals = [m[k] for m in metrics_all]
        summary[k] = {
            'mean': np.mean(vals),
            'std':  np.std(vals),
            'median': np.median(vals),
            'min': np.min(vals),
            'max': np.max(vals),
        }

    # 总 TP/FP/FN/TN
    total_tp = sum(m['tp'] for m in metrics_all)
    total_fp = sum(m['fp'] for m in metrics_all)
    total_fn = sum(m['fn'] for m in metrics_all)
    total_tn = sum(m['tn'] for m in metrics_all)
    eps = 1e-9
    overall_precision = total_tp / (total_tp + total_fp + eps)
    overall_recall    = total_tp / (total_tp + total_fn + eps)
    overall_f1        = 2 * overall_precision * overall_recall / (overall_precision + overall_recall + eps)
    overall_iou       = total_tp / (total_tp + total_fp + total_fn + eps)

    # ------- 打印汇总表 -------
    print("\n" + "=" * 70)
    print("                        评估结果汇总")
    print("=" * 70)
    print(f"{'指标':<14} {'均值':>8} {'标准差':>8} {'中位数':>8} {'最小值':>8} {'最大值':>8}")
    print("-" * 70)
    for k in keys:
        s = summary[k]
        print(f"{k.upper():<14} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['median']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}")
    print("-" * 70)
    print(f"\n总体统计（所有帧汇总）：")
    print(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn}")
    print(f"  总体 Precision = {overall_precision:.4f}")
    print(f"  总体 Recall    = {overall_recall:.4f}")
    print(f"  总体 F1-Score  = {overall_f1:.4f}")
    print(f"  总体 IoU       = {overall_iou:.4f}")
    print("=" * 70)

    return metrics_all, frame_nums, summary, \
           (total_tp, total_fp, total_fn, total_tn,
            overall_precision, overall_recall, overall_f1, overall_iou)


# ============================================================
# 3. 可视化
# ============================================================

def plot_results(metrics_all, frame_nums, summary, overall, args):
    out = args.output_dir

    precisions = [m['precision'] for m in metrics_all]
    recalls    = [m['recall']    for m in metrics_all]
    f1s        = [m['f1']        for m in metrics_all]
    ious       = [m['iou']       for m in metrics_all]
    accs       = [m['accuracy']  for m in metrics_all]

    total_tp, total_fp, total_fn, total_tn, op, or_, of1, oiou = overall

    # 设置中文字体（fallback 到英文）
    plt.rcParams['font.family'] = 'sans-serif'
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    except:
        pass
    plt.rcParams['axes.unicode_minus'] = False

    # ------ 图 1: 逐帧指标曲线 ------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    ax = axes[0]
    ax.plot(frame_nums, ious,  'b-', alpha=0.7, linewidth=0.8, label='IoU')
    ax.plot(frame_nums, f1s,  'r-', alpha=0.7, linewidth=0.8, label='F1-Score')
    ax.axhline(y=np.mean(ious), color='b', linestyle='--', alpha=0.5, label=f'IoU mean={np.mean(ious):.3f}')
    ax.axhline(y=np.mean(f1s),  color='r', linestyle='--', alpha=0.5, label=f'F1  mean={np.mean(f1s):.3f}')
    ax.set_ylabel('Score')
    ax.set_title('Per-Frame IoU & F1-Score')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(frame_nums, precisions, 'g-', alpha=0.7, linewidth=0.8, label='Precision')
    ax.plot(frame_nums, recalls,   'orange', alpha=0.7, linewidth=0.8, label='Recall')
    ax.axhline(y=np.mean(precisions), color='g', linestyle='--', alpha=0.5, label=f'P mean={np.mean(precisions):.3f}')
    ax.axhline(y=np.mean(recalls),  color='orange', linestyle='--', alpha=0.5, label=f'R mean={np.mean(recalls):.3f}')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Score')
    ax.set_title('Per-Frame Precision & Recall')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))

    plt.tight_layout()
    plt.savefig(os.path.join(out, '01_per_frame_metrics.png'), dpi=150)
    plt.close()
    print("  保存: 01_per_frame_metrics.png")

    # ------ 图 2: 指标分布直方图 ------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    dist_data = [
        (ious,  'IoU', 'blue'),
        (f1s,  'F1-Score', 'red'),
        (precisions, 'Precision', 'green'),
        (recalls, 'Recall', 'orange'),
        (accs,  'Accuracy', 'purple'),
    ]
    for idx, (data, name, color) in enumerate(dist_data):
        r, c = divmod(idx, 3)
        ax = axes[r][c]
        ax.hist(data, bins=40, color=color, alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.axvline(x=np.mean(data), color='black', linestyle='--', linewidth=1.5,
                   label=f'Mean={np.mean(data):.3f}')
        ax.set_title(f'{name} Distribution')
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)

    # 第 6 个位置放概览统计
    ax = axes[1][2]
    ax.axis('off')
    summary_text = (
        f"===== OVERALL SUMMARY =====\n\n"
        f"Frames evaluated: {len(metrics_all)}\n\n"
        f"--- Per-Frame Means ---\n"
        f"IoU:       {np.mean(ious):.4f} +/- {np.std(ious):.4f}\n"
        f"F1:        {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}\n"
        f"Precision: {np.mean(precisions):.4f} +/- {np.std(precisions):.4f}\n"
        f"Recall:    {np.mean(recalls):.4f} +/- {np.std(recalls):.4f}\n"
        f"Accuracy:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}\n\n"
        f"--- Overall (summed) ---\n"
        f"Precision: {op:.4f}\n"
        f"Recall:    {or_:.4f}\n"
        f"F1-Score:  {of1:.4f}\n"
        f"IoU:       {oiou:.4f}\n\n"
        f"TP={total_tp}  FP={total_fp}\n"
        f"FN={total_fn}  TN={total_tn}"
    )
    ax.text(0.05, 0.5, summary_text, transform=ax.transAxes,
            fontsize=9, fontfamily='monospace', verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out, '02_metric_distributions.png'), dpi=150)
    plt.close()
    print("  保存: 02_metric_distributions.png")

    # ------ 图 3: 逐帧 TP/FP/FN 堆叠面积图 ------
    fig, ax = plt.subplots(figsize=(16, 5))
    tps = [m['tp'] for m in metrics_all]
    fps = [m['fp'] for m in metrics_all]
    fns = [m['fn'] for m in metrics_all]

    ax.fill_between(frame_nums, 0, tps, alpha=0.7, color='green', label='TP (correct foreground)')
    ax.fill_between(frame_nums, tps, np.array(tps) + np.array(fps),
                    alpha=0.7, color='red', label='FP (false alarm)')
    ax.fill_between(frame_nums, tps, np.array(tps) + np.array(fns),
                    alpha=0.5, color='blue', label='FN (missed)')
    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Pixel Count')
    ax.set_title('Per-Frame TP / FP / FN Stacked Area')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=15))
    plt.tight_layout()
    plt.savefig(os.path.join(out, '03_tp_fp_fn_stack.png'), dpi=150)
    plt.close()
    print("  保存: 03_tp_fp_fn_stack.png")

    # ------ 图 4: 混淆矩阵 ------
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[total_tp, total_fp],
                   [total_fn, total_tn]])
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred FG', 'Pred BG'])
    ax.set_yticklabels(['GT FG', 'GT BG'])
    ax.set_title('Aggregate Confusion Matrix')
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm.sum() * 100
            ax.text(j, i, f'{val:,}\n({pct:.1f}%)',
                    ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='white' if val > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(out, '04_confusion_matrix.png'), dpi=150)
    plt.close()
    print("  保存: 04_confusion_matrix.png")

    # ------ 图 5: 不同阈值下的指标变化（鲁棒性分析）------
    # 这里我们不做阈值扫描，而是展示 IoU 的 CDF（累计分布）
    fig, ax = plt.subplots(figsize=(10, 5))
    ious_sorted = np.sort(ious)
    cdf = np.arange(1, len(ious_sorted) + 1) / len(ious_sorted)
    ax.plot(ious_sorted, cdf, 'b-', linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median')
    ax.axvline(x=np.median(ious), color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('IoU')
    ax.set_ylabel('CDF')
    ax.set_title(f'IoU Cumulative Distribution (Median={np.median(ious):.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(out, '05_iou_cdf.png'), dpi=150)
    plt.close()
    print("  保存: 05_iou_cdf.png")

    # ------ 图 6: 样本帧可视化 (最好/最差/中位) ------
    best_idx  = np.argmax(ious)
    worst_idx = np.argmin(ious)
    median_idx = np.argsort(ious)[len(ious) // 2]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    titles = ['Best Frame', 'Median Frame', 'Worst Frame']
    idxes  = [best_idx, median_idx, worst_idx]

    for col, (title, idx) in enumerate(zip(titles, idxes)):
        fn = frame_nums[idx]
        m  = metrics_all[idx]

        # 读取原图、预测、真值
        pred_path = os.path.join(
            args.pred_dir, f"{args.prefix_pred}{fn - args.init:05d}.png")
        gt_path = os.path.join(
            args.gt_dir, f"{args.prefix_gt}{fn:06d}{args.gt_ext}")

        pred = load_mask(pred_path)
        gt   = load_mask(gt_path)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        # 原图（尝试加载 input 目录的图像）
        input_path = os.path.join(
            os.path.dirname(args.gt_dir).replace('groundtruth', ''),
            'input', f'in{fn:06d}.jpg')
        orig = cv2.imread(input_path)
        if orig is not None:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

        # GT Mask
        axes[0][col].imshow(gt, cmap='gray')
        axes[0][col].set_title(f'{title} (#{fn})\nGT Mask')
        axes[0][col].axis('off')

        # Pred Mask
        axes[1][col].imshow(pred, cmap='gray')
        axes[1][col].set_title(f'Pred Mask\nIoU={m["iou"]:.3f} F1={m["f1"]:.3f}')
        axes[1][col].axis('off')

        # TP/FP/FN 彩色图
        pred_bin = (pred > 127).astype(np.uint8)
        gt_bin   = (gt   > 127).astype(np.uint8)
        vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        vis[(pred_bin == 1) & (gt_bin == 1)] = [0, 255, 0]    # TP: green
        vis[(pred_bin == 1) & (gt_bin == 0)] = [255, 0, 0]    # FP: red
        vis[(pred_bin == 0) & (gt_bin == 1)] = [0, 0, 255]    # FN: blue
        # TN (background) stays black
        axes[2][col].imshow(vis)
        axes[2][col].set_title('Green=TP  Red=FP  Blue=FN')
        axes[2][col].axis('off')

    plt.suptitle('Sample Frame Visualizations: Best / Median / Worst IoU',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out, '06_sample_frames.png'), dpi=150)
    plt.close()
    print("  保存: 06_sample_frames.png")

    # ================================================================
    # 保存逐帧数据到 CSV
    # ================================================================
    csv_path = os.path.join(out, 'per_frame_metrics.csv')
    with open(csv_path, 'w') as f:
        header = 'frame,tp,fp,fn,tn,precision,recall,f1,iou,accuracy\n'
        f.write(header)
        for m in metrics_all:
            f.write(f"{m['frame']},{m['tp']},{m['fp']},{m['fn']},{m['tn']},"
                    f"{m['precision']:.6f},{m['recall']:.6f},{m['f1']:.6f},"
                    f"{m['iou']:.6f},{m['accuracy']:.6f}\n")
    print(f"  保存: per_frame_metrics.csv ({len(metrics_all)} rows)")

    # 保存汇总到 TXT
    summary_path = os.path.join(out, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("运动目标检测评估报告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"评估帧数: {len(metrics_all)}\n")
        f.write(f"帧范围:   {frame_nums[0]} ~ {frame_nums[-1]}\n\n")
        f.write("--- Per-Frame Statistics ---\n")
        f.write(f"{'Metric':<12} {'Mean':>8} {'Std':>8} {'Median':>8} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 60 + "\n")
        for k in ['iou', 'f1', 'precision', 'recall', 'accuracy']:
            s = summary[k]
            f.write(f"{k.upper():<12} {s['mean']:>8.4f} {s['std']:>8.4f} "
                    f"{s['median']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}\n")
        f.write("\n--- Overall (Aggregated) ---\n")
        f.write(f"TP={total_tp}  FP={total_fp}  FN={total_fn}  TN={total_tn}\n")
        f.write(f"Precision = {op:.4f}\n")
        f.write(f"Recall    = {or_:.4f}\n")
        f.write(f"F1-Score  = {of1:.4f}\n")
        f.write(f"IoU       = {oiou:.4f}\n")
    print(f"  保存: summary.txt")

    # ------ 图 7: 随机三帧 Mask 可视化 ------
    random_viz(metrics_all, frame_nums, args, out)

    print(f"\n全部评估图表已保存至: {os.path.abspath(out)}")


def random_viz(metrics_all, frame_nums, args, out):
    """随机选取 3 帧，并排展示: 原图 | GT Mask | Pred Mask | 差异图"""
    rng = np.random.RandomState(42)
    picks = rng.choice(len(metrics_all), size=min(3, len(metrics_all)), replace=False)

    fig, axes = plt.subplots(3, 5, figsize=(20, 12))

    col_titles = ['Original Frame', 'GT Mask', 'Predicted Mask',
                  'Difference Map', 'Overlay (Green=Pred)']

    # 尝试加载原图路径
    for row_idx, pick in enumerate(picks):
        fn = frame_nums[pick]
        m = metrics_all[pick]

        # 文件路径
        pred_path = os.path.join(
            args.pred_dir, f"{args.prefix_pred}{fn - args.init:05d}.png")
        gt_path = os.path.join(
            args.gt_dir, f"{args.prefix_gt}{fn:06d}{args.gt_ext}")

        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt   = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if pred is None or gt is None:
            continue
        if pred.shape != gt.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        # 原图
        input_dir = os.path.join(
            os.path.dirname(os.path.dirname(args.gt_dir)), 'input')
        orig = None
        for ext in ['.jpg', '.png', '.bmp']:
            p = os.path.join(input_dir, f'in{fn:06d}{ext}')
            if os.path.exists(p):
                orig = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
                break

        # 差异图: Red=FP, Blue=FN, Green=TP
        pred_bin = (pred > 127).astype(np.uint8)
        gt_bin   = (gt   > 127).astype(np.uint8)
        diff_map = np.zeros((*gt.shape, 3), dtype=np.uint8)
        diff_map[(pred_bin == 1) & (gt_bin == 1)] = [0, 255, 0]    # TP: green
        diff_map[(pred_bin == 1) & (gt_bin == 0)] = [255, 0, 0]    # FP: red
        diff_map[(pred_bin == 0) & (gt_bin == 1)] = [0, 0, 255]    # FN: blue

        # Overlay: 原图 + 绿色预测轮廓
        if orig is not None:
            overlay = orig.copy()
            overlay[pred_bin == 1] = (overlay[pred_bin == 1] * 0.5 +
                                      np.array([0, 255, 0]) * 0.5).astype(np.uint8)
        else:
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[pred_bin == 1] = [0, 255, 0]

        # 填充这行
        row_imgs = [orig, gt, pred, diff_map, overlay]
        for col_idx, img in enumerate(row_imgs):
            ax = axes[row_idx][col_idx]
            if img is not None:
                if len(img.shape) == 2:
                    ax.imshow(img, cmap='gray')
                else:
                    ax.imshow(img)
            if col_idx == 0:
                ax.set_ylabel(f'Frame #{fn}\nIoU={m["iou"]:.3f} F1={m["f1"]:.3f}',
                              fontsize=10, fontweight='bold')
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11, fontweight='bold')
            ax.axis('off')

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green',  alpha=0.5, label='TP (Correct)'),
        Patch(facecolor='red',    alpha=0.5, label='FP (False Alarm)'),
        Patch(facecolor='blue',   alpha=0.5, label='FN (Missed)'),
        Patch(facecolor='black',  alpha=0.5, label='TN (Background)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=10, framealpha=0.8)

    plt.suptitle('Random 3-Frame Mask Visualization: Original | GT | Predicted | Error Map | Overlay',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(os.path.join(out, '07_random_frame_viz.png'), dpi=150)
    plt.close()
    print("  保存: 07_random_frame_viz.png")

if __name__ == '__main__':
    args = parse_args()
    metrics_all, frame_nums, summary, overall = run_evaluation(args)
    plot_results(metrics_all, frame_nums, summary, overall, args)
