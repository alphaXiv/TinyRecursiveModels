
import re
import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


# Regexes: allow simple markdown bold wrappers, optional parentheticals, and unicode ± variants
CLAIM_RE = re.compile(r"\*{0,2}\s*Paper'?s\s+Claim(?:\s*\([^)]*\))?\*{0,2}\s*[:\-–—]?\s*([0-9]+(?:\.[0-9]+)?)\s*%", flags=re.IGNORECASE)
ACHIEVED_RE = re.compile(r"\*{0,2}\s*Achieved\s+Result(?:\s*\([^)]*\))?\*{0,2}\s*[:\-–—]?\s*([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:±|\u00B1|\+/-)\s*([0-9]+(?:\.[0-9]+)?)\s*%?", flags=re.IGNORECASE)
ACHIEVED_SIMPLE_RE = re.compile(r"\*{0,2}\s*Achieved\s+Result(?:\s*\([^)]*\))?\*{0,2}\s*[:\-–—]?\s*([0-9]+(?:\.[0-9]+)?)\s*%", flags=re.IGNORECASE)


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def find_nearest_heading(lines: List[str], idx: int) -> str:
    # search upwards for a heading (##, ###) or a short preceding titled line
    for i in range(idx - 1, max(-1, idx - 40), -1):
        line = lines[i].strip()
        if line.startswith("##") or line.startswith("###") or line.startswith("#"):
            return line.lstrip('#').strip()
        m = re.match(r"^[-*]\s*\*\*(.+)\*\*", line)
        if m:
            return m.group(1).strip()
        if 0 < len(line) < 60 and line == line.upper():
            return line.strip()
    return "Unknown"


def parse_files(paths: List[str]) -> List[Tuple[str, float, float, Optional[float]]]:
    """Return list of tuples: (label, claimed, achieved, achieved_err)"""
    items = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        text = open(path, 'r', encoding='utf-8').read()

        # find claims and achieved with character offsets so we can pair nearby ones
        claims = [(m.start(), float(m.group(1))) for m in CLAIM_RE.finditer(text)]
        achieveds = []
        for m in ACHIEVED_RE.finditer(text):
            achieveds.append((m.start(), float(m.group(1)), float(m.group(2))))
        for m in ACHIEVED_SIMPLE_RE.finditer(text):
            # avoid duplicating entries already captured by ACHIEVED_RE
            if not any(abs(m.start() - a[0]) < 4 for a in achieveds):
                achieveds.append((m.start(), float(m.group(1)), None))

        if not claims:
            continue

        if not achieveds:
            print(f"Found {len(claims)} claim(s) in {path} but no achieved matches; skipping.")
            continue

        achieveds.sort(key=lambda x: x[0])
        lines = text.splitlines()

        print("Found achieveds:", achieveds)
        for cpos, cval in claims:
            # pick the achieved entry with accuracy closest to the claim value
            after = [a for a in achieveds if a[0] >= cpos]
            if after:
                apos, aval, aerr = min(after, key=lambda x: abs(x[1] - cval))
            else:
                apos, aval, aerr = min(achieveds, key=lambda x: abs(x[1] - cval))

            # map character offset to line index for labeling
            cum = 0
            line_idx = 0
            for idx, L in enumerate(lines):
                cum += len(L) + 1
                if cum > cpos:
                    line_idx = idx
                    break
            label = find_nearest_heading(lines, line_idx)
            items.append((label, cval, aval, aerr))

    return items


def plot_combined(items, outpath: str):
    labels = [t[0] for t in items]
    claimed = np.array([t[1] for t in items])
    achieved = np.array([t[2] for t in items])
    errs = np.array([t[3] if t[3] is not None else 0.0 for t in items])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    ax.bar(x - width / 2, claimed, width, label='Paper Claim', color='#4C72B0')
    ax.bar(x + width / 2, achieved, width, yerr=errs, label='Achieved (±SE)', color='#DD8452', capsize=5)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Paper Claim vs Achieved Result')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0, max(100, float(np.max(claimed)) + 10, float(np.max(achieved)) + 10))
    plt.tight_layout()
    fig.savefig(outpath)
    print(f"Saved combined plot to {outpath}")


def plot_per_item(items, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for label, claimed, achieved, err in items:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([0], [claimed], width=0.4, label='Paper Claim', color='#4C72B0')
        ax.bar([1], [achieved], width=0.4, yerr=[err or 0.0], label='Achieved', color='#DD8452', capsize=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Paper Claim', 'Achieved'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(label)
        ax.set_ylim(0, max(100, claimed + 10, achieved + 10))
        for i, v in enumerate([claimed, achieved]):
            ax.text(i, v + 1, f"{v:.2f}%", ha='center')
        plt.tight_layout()
        fname = os.path.join(outdir, f"claims_vs_achieved_{slugify(label)}.png")
        fig.savefig(fname)
        plt.close(fig)
        print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', default=['REPORT.md'], help='Markdown files to parse')
    parser.add_argument('--outdir', default='reports/plots', help='Output directory for plots')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    args = parser.parse_args()

    items = parse_files(args.files)
    if not items:
        print('No claim/achieved pairs found in the provided files.')
        return

    os.makedirs(args.outdir, exist_ok=True)
    combined_path = os.path.join(args.outdir, 'claims_vs_achieved_combined.png')
    plot_combined(items, combined_path)
    plot_per_item(items, args.outdir)

    if args.show:
        img = plt.imread(combined_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
import matplotlib.pyplot as plt
