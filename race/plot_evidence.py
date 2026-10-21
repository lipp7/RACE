"""Make a stacked single-column private/shared and pixel-intervention figure."""
import argparse
import numpy as np
from PIL import Image
from maft.data import CUBImages
from .io import project_root, relative_path, read_json
from .evidence import crop_rectangles


def projection(weights, rectangles, width, height):
    heat = np.zeros((height, width), dtype=np.float32)
    for value, (left, top, right, bottom) in zip(weights, rectangles):
        left, right = np.clip([int(np.floor(left)), int(np.ceil(right))], 0, width)
        top, bottom = np.clip([int(np.floor(top)), int(np.ceil(bottom))], 0, height)
        if right > left and bottom > top:
            heat[top:bottom, left:right] += value / ((right - left) * (bottom - top))
    return heat / max(float(heat.max()), 1e-20)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pixels", required=True, help="Pixel analysis summary.json")
    parser.add_argument("--cub-root", default="data/CUB_200_2011")
    parser.add_argument("--output", default="figures/evidence")
    args = parser.parse_args()
    project_root()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": .7,
                         "pdf.fonttype": 42, "svg.fonttype": "path"})
    source = CUBImages(relative_path(args.cub_root), "test", None, view_grid=3)
    with np.load(relative_path(args.weights)) as data:
        weights = {key: data[key].copy() for key in ("private", "shared", "labels")}
    summary = read_json(relative_path(args.pixels))
    figure = plt.figure(figsize=(3.5, 3.75))
    figure.text(.02, .985, "(a) Private vs. Shared Evidence", va="top")
    grid = figure.add_gridspec(2, 3, wspace=.035, hspace=.045, left=.085, right=.99, top=.93, bottom=.43)
    for column, category in enumerate((0, 75, 150)):
        index = int(np.flatnonzero(weights["labels"] == category)[0])
        with Image.open(source.items[index][0]) as opened:
            image = np.asarray(opened.convert("RGB"))
        height, width = image.shape[:2]
        side = min(height, width)
        y, x = (height - side) // 2, (width - side) // 2
        for row, key in enumerate(("private", "shared")):
            axis = figure.add_subplot(grid[row, column])
            heat = projection(weights[key][index], crop_rectangles(width, height), width, height)
            axis.imshow(image[y:y + side, x:x + side])
            axis.imshow(heat[y:y + side, x:x + side], cmap="inferno", alpha=.55, vmin=0, vmax=1)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(key.capitalize())
    figure.text(.02, .395, "(b) Pixel interventions", va="top")
    axis = figure.add_axes([.18, .075, .79, .24])
    for offset, (policy, label, color) in enumerate([
            ("private", "Private", "#c0392b"), ("shared", "Shared", "#2e86c1"),
            ("random_crop", "Random", "#95a5a6")]):
        means, errors = [], []
        for kind in ("delete", "retain"):
            group = next(g for g in summary["groups"] if g["policy"] == policy and g["intervention"] == kind and abs(g["fraction"] - .2) < 1e-9)
            values = [r["margin_drop"] for r in group["per_run"]]
            means.append(np.mean(values))
            errors.append(np.std(values, ddof=1) if len(values) > 1 else 0.)
        axis.bar(np.arange(2) + (offset - 1) * .23, means, .23, yerr=errors, color=color,
                 label=label, error_kw=dict(lw=.8, capsize=3))
    axis.axhline(0, color="black", lw=.5)
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["Delete", "Retain only"])
    axis.set_ylabel("Margin decrease")
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 1.01),
                handlelength=1, columnspacing=1.1)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    output = relative_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        figure.savefig(output.with_suffix("." + suffix), dpi=600)
    plt.close(figure)


if __name__ == "__main__":
    main()
