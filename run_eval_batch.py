#!/usr/bin/env python3
"""
批量评估流水线
-------------
对多个 CDnet 2014 视频序列运行 motion_detect.exe 并调用 evaluate.py 生成报告。
"""

import subprocess
import os
import sys
import argparse
import csv
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).parent.resolve()
# Python with cv2 for evaluate.py
EVAL_PYTHON = r"C:\Users\29593\anaconda3\envs\deeplearning\python.exe"
BUILD_DIR = ROOT / "build"
EXE = BUILD_DIR / "motion_detect.exe"
DATASET = ROOT / "dataset"
RESULTS_DIR = ROOT / "evaluation_results"

# ============================================================
# CDnet 2014 全部序列及其 temporal ROI
# 格式: (类别, 序列名, 帧数, temporal_roi_start, temporal_roi_end)
# ============================================================
CDNET_SEQUENCES = [
    # === baseline (4 sequences) ===
    ("baseline", "highway", 1700, 470, 1700, True),
    ("baseline", "office", 2050, 570, 2050, True),
    ("baseline", "pedestrians", 1099, 300, 1099, True),
    ("baseline", "PETS2006", 1200, 300, 1200, True),

    # === dynamicBackground (6 sequences) ===
    ("dynamicBackground", "boats", 7999, 1900, 7999, True),
    ("dynamicBackground", "canoe", 1189, 400, 1189, True),
    ("dynamicBackground", "fall", 4000, 1000, 4000, True),
    ("dynamicBackground", "fountain01", 1184, 400, 1184, True),
    ("dynamicBackground", "fountain02", 1499, 500, 1499, True),
    ("dynamicBackground", "overpass", 3000, 900, 3000, True),

    # === cameraJitter (4 sequences) ===
    ("cameraJitter", "badminton", 1150, 300, 1150, True),
    ("cameraJitter", "boulevard", 2500, 790, 2500, True),
    ("cameraJitter", "sidewalk", 1200, 300, 1200, True),
    ("cameraJitter", "traffic", 1570, 470, 1570, True),

    # === shadow (6 sequences) ===
    ("shadow", "backdoor", 2000, 470, 2000, True),
    ("shadow", "bungalows", 1700, 470, 1700, True),
    ("shadow", "busStation", 1100, 330, 1100, True),
    ("shadow", "copyMachine", 3400, 1000, 3400, True),
    ("shadow", "cubicle", 7400, 2200, 7400, True),
    ("shadow", "peopleInShade", 1199, 400, 1199, True),

    # === thermal (5 sequences) ===
    ("thermal", "corridor", 6000, 700, 6000, True),
    ("thermal", "diningRoom", 4200, 700, 4200, True),
    ("thermal", "lakeSide", 6500, 700, 6500, True),
    ("thermal", "library", 5400, 700, 5400, True),
    ("thermal", "park", 600, 160, 600, True),
]


def read_temporal_roi(cat, seq):
    """从 temporalROI.txt 读取 ROI，失败则返回 None"""
    roi_file = DATASET / cat / seq / "temporalROI.txt"
    if roi_file.exists():
        with open(roi_file) as f:
            parts = f.read().strip().split()
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    return None

def parse_args():
    parser = argparse.ArgumentParser(description="批量评估流水线")
    parser.add_argument("--categories", nargs="+", default=["all"],
                        help="要评估的类别 (default: all)")
    parser.add_argument("--sequences", nargs="+", default=[],
                        help="指定序列名（category/seqname）")
    parser.add_argument("--init", type=int, default=25,
                        help="模型初始化帧数 (ViBe N, default: 25)")
    parser.add_argument("--skip-masks", action="store_true",
                        help="跳过 Mask 生成（仅运行评估）")
    parser.add_argument("--vibe-r", type=int, default=25)
    parser.add_argument("--vibe-min", type=int, default=2)
    parser.add_argument("--vibe-phi", type=int, default=20)
    parser.add_argument("--thresh-lambda", type=float, default=5.0,
                        help="自适应阈值倍数")
    parser.add_argument("--extra-params", type=str, default="",
                        help="额外参数（空格分隔的key value对，如 '--gf-thresh 0.4 --flow-w 0.8'）")
    return parser.parse_args()


def run_cmd(cmd, desc=""):
    """运行命令并打印输出"""
    print(f"  [{desc}] {' '.join(cmd)}")
    log_file = str(BUILD_DIR / f"_batch_log_{desc.replace('/', '_')}.txt")
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    timeout=7200, cwd=str(BUILD_DIR))
        if result.returncode != 0:
            # 读取最后几行查看错误
            with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")
            print(f"  ERROR (exit={result.returncode}), log: {log_file}")
            return False
        # 读取最后几行输出
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        for line in lines[-3:]:
            print(f"    {line.rstrip()}")
        return True
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 7200s")
        return False
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        return False


def run_sequence(cat, seq, num_frames, roi_start, roi_end, args):
    """对单个序列运行完整评估流水线"""
    name = f"{cat}/{seq}"

    # 优先从 temporalROI.txt 读取
    roi = read_temporal_roi(cat, seq)
    if roi:
        roi_start, roi_end = roi

    print(f"\n{'='*60}")
    print(f"  处理: {name} ({num_frames} frames, ROI {roi_start}-{roi_end})")
    print(f"{'='*60}")

    input_pattern = str((DATASET / cat / seq / "input" / "in%06d.jpg").resolve())
    gt_dir = str((DATASET / cat / seq / "groundtruth").resolve())
    out_dir = str((ROOT / f"eval_out_{cat}_{seq}").resolve())

    # --- Step 1: 生成 Mask ---
    if not args.skip_masks:
        print(f"  [1/2] 生成 Mask -> {out_dir}")
        cmd = [
            str(EXE),
            input_pattern,
            "--batch",
            "--output-dir", out_dir,
            "--save-every", "1",
            "--init", str(args.init),
            "--vibe-r", str(args.vibe_r),
            "--vibe-min", str(args.vibe_min),
            "--vibe-phi", str(args.vibe_phi),
            "--lambda", str(args.thresh_lambda),
        ]
        # 解析并添加额外参数
        if args.extra_params:
            extra = args.extra_params.split()
            cmd.extend(extra)

        # 清理旧的输出
        if os.path.exists(out_dir):
            import shutil
            shutil.rmtree(out_dir)

        if not run_cmd(cmd, desc=name):
            print(f"  FAILED: Mask 生成失败")
            return None

    # --- Step 2: 评估 ---
    eval_out = str(RESULTS_DIR / f"eval_{cat}_{seq}")
    print(f"  [2/2] 评估 -> {eval_out}")

    eval_cmd = [
        EVAL_PYTHON,
        str(ROOT / "evaluate.py"),
        out_dir,  # pred_dir
        gt_dir,    # gt_dir
        "--temporal-roi", str(roi_start), str(roi_end),
        "--init", str(args.init),
        "--output-dir", eval_out,
    ]
    if not run_cmd(eval_cmd, desc=f"eval {name}"):
        print(f"  WARNING: 评估可能不完整")

    # 读取汇总指标
    summary_txt = os.path.join(eval_out, "summary.txt")
    if os.path.exists(summary_txt):
        with open(summary_txt, 'r') as f:
            content = f.read()
        # 解析 Overall F1, IoU 等
        result = {
            "category": cat,
            "sequence": seq,
            "num_frames": num_frames,
            "roi": f"{roi_start}-{roi_end}",
        }
        # 解析 Overall 汇总
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Precision ='):
                result['precision'] = float(line.split('=')[-1].strip())
            if line.startswith('Recall    ='):
                result['recall'] = float(line.split('=')[-1].strip())
            if line.startswith('F1-Score  ='):
                result['f1'] = float(line.split('=')[-1].strip())
            if line.startswith('IoU       ='):
                result['iou'] = float(line.split('=')[-1].strip())
        print(f"  -> F1={result.get('f1', 'NA')}, IoU={result.get('iou', 'NA')}, "
              f"P={result.get('precision', 'NA')}, R={result.get('recall', 'NA')}")
        return result
    else:
        print(f"  WARNING: 找不到 summary.txt")
        return None


def main():
    args = parse_args()

    # 筛选序列
    selected = []
    cat_filter = args.categories if "all" not in args.categories else None
    seq_filter = args.sequences

    for cat, seq, nf, rs, re, _ in CDNET_SEQUENCES:
        if cat_filter and cat not in cat_filter:
            continue
        if seq_filter and f"{cat}/{seq}" not in seq_filter and seq not in seq_filter:
            continue
        selected.append((cat, seq, nf, rs, re))

    print(f"待评估: {len(selected)} 个序列")
    print(f"模型初始化帧数: {args.init}")
    print(f"参数: vibeR={args.vibe_r} vibeMin={args.vibe_min} "
          f"vibePhi={args.vibe_phi} lambda={args.thresh_lambda}"
          f"{' extra=' + args.extra_params if args.extra_params else ''}")

    results = []
    for cat, seq, nf, rs, re in selected:
        r = run_sequence(cat, seq, nf, rs, re, args)
        if r:
            results.append(r)

    # 汇总报告
    print("\n\n" + "=" * 80)
    print("                        批量评估汇总")
    print("=" * 80)
    print(f"{'序列':<35} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8}")
    print("-" * 80)

    # 按类别分组
    by_cat = {}
    for r in results:
        cat = r['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(r)

    for cat in sorted(by_cat.keys()):
        seqs = by_cat[cat]
        for r in seqs:
            name = f"{r['category']}/{r['sequence']}"
            f1 = r.get('f1', 0)
            iou = r.get('iou', 0)
            prec = r.get('precision', 0)
            rec = r.get('recall', 0)
            print(f"{name:<35} {f1:>8.4f} {iou:>8.4f} {prec:>8.4f} {rec:>8.4f}")
        if len([s for s in CDNET_SEQUENCES if s[0] == cat]) > 1:
            avg_f1 = sum(r.get('f1', 0) for r in seqs) / len(seqs)
            avg_iou = sum(r.get('iou', 0) for r in seqs) / len(seqs)
            print(f"  {cat} 均值{'':<24} {avg_f1:>8.4f} {avg_iou:>8.4f}")

    # 保存 CSV
    csv_path = ROOT / "eval_batch_results.csv"
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果已保存到: {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()
