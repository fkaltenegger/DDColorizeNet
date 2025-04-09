import os
import sys
import time
import cv2
import lpips
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.morphology import dilation, disk
import torch.nn.functional as F
from matplotlib import gridspec
from skimage.segmentation import slic


def save_metric_tables(df, metric, image_dir, type, lvl="lvl2", original_dir="img_orig"):
    def create_table(df_subset, title, filename):
        df_sorted = df_subset.sort_values(by='Image')
        num_rows = len(df_sorted)
        fig = plt.figure(figsize=(13, num_rows * 2.8))
        outer_grid = gridspec.GridSpec(num_rows, 1, wspace=0.2, hspace=0.4)

        for row_idx, (_, row) in enumerate(df_sorted.iterrows()):
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                1,
                5,
                subplot_spec=outer_grid[row_idx],
                width_ratios=[1, 1, 3, 1.5, 3]
            )

            if type == "c":
                filename_img = f"{row['Image']}_{row['Seed']}.png"
            else:
                filename_img = f"{row['Image']}_{lvl}_{row['Seed']}.png"

            image_path = os.path.join(image_dir, filename_img)
            gt_path = os.path.join(original_dir, f"{row['Image']}.png")

            if not os.path.exists(image_path) or not os.path.exists(gt_path):
                continue

            pred = cv2.imread(image_path)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

            pred = cv2.resize(pred, (256, 256))
            gt = cv2.resize(gt, (256, 256))

            edge_mask = dilation((gt.mean(axis=2) < 250), disk(1))
            delta_e_map = deltaE_ciede2000(rgb2lab(gt / 255.0), rgb2lab(pred / 255.0))
            delta_e_bleed = np.full_like(delta_e_map, np.nan)
            delta_e_bleed[edge_mask] = delta_e_map[edge_mask]

            ax0 = plt.Subplot(fig, inner_grid[0])
            ax0.axis('off')
            ax0.text(0.5, 0.5, str(row['Image']), ha='center', va='center', fontsize=11)
            fig.add_subplot(ax0)

            ax1 = plt.Subplot(fig, inner_grid[1])
            ax1.axis('off')
            ax1.text(0.5, 0.5, str(row['Seed']), ha='center', va='center', fontsize=11)
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner_grid[2])
            ax2.imshow(pred)
            ax2.axis('off')
            fig.add_subplot(ax2)

            ax3 = plt.Subplot(fig, inner_grid[3])
            ax3.axis('off')
            ax3.text(0.5, 0.5, f"{row[metric]:.4f}", ha='center', va='center', fontsize=11)
            fig.add_subplot(ax3)

            ax4 = plt.Subplot(fig, inner_grid[4])
            ax4.imshow(delta_e_bleed, cmap='inferno')
            ax4.axis('off')
            fig.add_subplot(ax4)

        fig.suptitle(f"{title} by {metric}", fontsize=16)
        plt.savefig(filename, dpi=200)
        plt.close()

    if metric == 'CDR':
        best_df = df.loc[df.groupby('Image')[metric].idxmax()]
        worst_df = df.loc[df.groupby('Image')[metric].idxmin()]
    else:
        best_df = df.loc[df.groupby('Image')[metric].idxmin()]
        worst_df = df.loc[df.groupby('Image')[metric].idxmax()]

    mean_df = df.groupby('Image').mean(numeric_only=True).reset_index()
    mean_seeds = df.groupby('Image')[metric].mean().reset_index().merge(
        df, on=['Image', metric], how='left'
    )

    create_table(best_df, "Best Images", f"table_best_{metric.lower()}_{type}.png")
    create_table(worst_df, "Worst Images", f"table_worst_{metric.lower()}_{type}.png")
    create_table(mean_seeds, "Closest to Mean", f"table_mean_{metric.lower()}_{type}.png")


def save_overall_metric_tables(df, type, image_dir, lvl="lvl2", original_dir="img_orig"):
    df['rank_deltaE'] = df.groupby('Image')['DeltaE'].rank(method='min', ascending=True)
    df['rank_CDR'] = df.groupby('Image')['CDR'].rank(method='min', ascending=False)
    df['Overall_Rank'] = df['rank_deltaE'] + df['rank_CDR']

    best_overall_df = df.loc[df.groupby('Image')['Overall_Rank'].idxmin()]
    worst_overall_df = df.loc[df.groupby('Image')['Overall_Rank'].idxmax()]

    def create_overall_table(df_subset, title, filename):
        df_sorted = df_subset.sort_values(by='Image')
        num_rows = len(df_sorted)

        fig = plt.figure(figsize=(15, num_rows * 3))
        outer_grid = gridspec.GridSpec(num_rows, 1, wspace=0.2, hspace=0.4)

        for row_idx, (_, row) in enumerate(df_sorted.iterrows()):
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                1,
                5,
                subplot_spec=outer_grid[row_idx],
                width_ratios=[1, 1, 3, 3, 3]
            )

            if type == "c":
                filename_img = f"{row['Image']}_{row['Seed']}.png"
            else:
                filename_img = f"{row['Image']}_{lvl}_{row['Seed']}.png"

            image_path = os.path.join(image_dir, filename_img)
            gt_path = os.path.join(original_dir, f"{row['Image']}.png")

            if not os.path.exists(image_path) or not os.path.exists(gt_path):
                continue

            pred = cv2.imread(image_path)
            pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
            gt = cv2.imread(gt_path)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

            pred = cv2.resize(pred, (256, 256))
            gt = cv2.resize(gt, (256, 256))

            edge_mask = dilation((gt.mean(axis=2) < 250), disk(1))
            delta_e_map = deltaE_ciede2000(rgb2lab(gt / 255.0), rgb2lab(pred / 255.0))
            delta_e_bleed = np.full_like(delta_e_map, np.nan)
            delta_e_bleed[edge_mask] = delta_e_map[edge_mask]

            ax0 = plt.Subplot(fig, inner_grid[0])
            ax0.axis('off')
            ax0.text(0.5, 0.5, str(row['Image']), ha='center', va='center', fontsize=11)
            fig.add_subplot(ax0)

            ax1 = plt.Subplot(fig, inner_grid[1])
            ax1.axis('off')
            ax1.text(0.5, 0.5, str(row['Seed']), ha='center', va='center', fontsize=11)
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner_grid[2])
            ax2.imshow(pred)
            ax2.axis('off')
            fig.add_subplot(ax2)

            summary_text = (
                f"LPIPS: {row['LPIPS']:.4f}\n"
                f"DeltaE: {row['DeltaE']:.4f}\n"
                f"CDR: {row['CDR']:.4f}\n"
                f"Overall: {row['Overall_Rank']:.4f}"
            )
            ax3 = plt.Subplot(fig, inner_grid[3])
            ax3.axis('off')
            ax3.text(0.5, 0.5, summary_text, ha='center', va='center', fontsize=10)
            fig.add_subplot(ax3)

            ax4 = plt.Subplot(fig, inner_grid[4])
            ax4.imshow(delta_e_bleed, cmap='inferno')
            ax4.axis('off')
            fig.add_subplot(ax4)

        fig.suptitle(title, fontsize=16)
        plt.savefig(filename, dpi=200)
        plt.close()

    create_overall_table(
        best_overall_df,
        "Overall Best Seeds (Lower LPIPS & DeltaE, Higher CDR)",
        f"table_overall_best_{type}.png"
    )
    create_overall_table(
        worst_overall_df,
        "Overall Worst Seeds (Higher LPIPS & DeltaE, Lower CDR)",
        f"table_overall_worst_{type}.png"
    )


def plot_metric_summary(df, value_column: str, plot_title: str, filename: str, color='mediumslateblue'):
    plt.figure(figsize=(10, 5))
    bars = plt.bar(df['Image'], df[value_column], color=color)
    plt.xlabel('Image')
    plt.ylabel(value_column)
    plt.title(plot_title)
    plt.xticks(rotation=45)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha='center',
            va='bottom',
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_median_scores(df, df_median, metric, output_path):
    plt.figure(figsize=(12, 6))
    images = df_median['Image'].tolist()
    data = [df[df['Image'] == img][metric].values for img in images]
    box = plt.boxplot(data, labels=images, patch_artist=True, showmeans=False)

    for patch in box['boxes']:
        patch.set_facecolor('mediumslateblue')

    for i, median in enumerate(df_median[metric]):
        plt.text(
            i + 1,
            median + 0.01,
            f"{median:.3f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color='black'
        )
    plt.xlabel('Image')
    plt.ylabel(f'{metric}')
    plt.title(f'{metric} Distribution per Image (with Medians)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def cdr(inputs, gt, kernel_size=5, target_region_mask=None, mask=None):
    if isinstance(inputs, np.ndarray):
        inputs = torch.Tensor(inputs)
    if isinstance(gt, np.ndarray):
        gt = torch.Tensor(gt)

    assert inputs.dim() == 3
    assert gt.dim() == 3

    inputs = inputs.permute(1, 2, 0)
    gt = gt.permute(1, 2, 0)

    pad_size = (kernel_size - 1, kernel_size - 1, kernel_size - 1, kernel_size - 1)

    gt_a = gt[..., 1:2].repeat(1, 1, 3)
    gt_b = gt[..., 2:3].repeat(1, 1, 3)

    inputs_a = inputs[..., 1:2].repeat(1, 1, 3)
    inputs_b = inputs[..., 2:3].repeat(1, 1, 3)

    gt_a_slic = torch.Tensor(
        slic(gt_a.double().numpy(), n_segments=250, compactness=10, sigma=1, start_label=1)
    )
    gt_b_slic = torch.Tensor(
        slic(gt_b.double().numpy(), n_segments=250, compactness=10, sigma=1, start_label=1)
    )
    inputs_a_slic = torch.Tensor(
        slic(inputs_a.double().numpy(), n_segments=250, compactness=10, sigma=1, start_label=1)
    )
    inputs_b_slic = torch.Tensor(
        slic(inputs_b.double().numpy(), n_segments=250, compactness=10, sigma=1, start_label=1)
    )

    gt_a_slic = F.pad(gt_a_slic, pad_size, "constant", 0)
    gt_b_slic = F.pad(gt_b_slic, pad_size, "constant", 0)
    inputs_a_slic = F.pad(inputs_a_slic, pad_size, "constant", 0)
    inputs_b_slic = F.pad(inputs_b_slic, pad_size, "constant", 0)

    cats_a = torch.Tensor(mask)
    cats_b = torch.Tensor(mask)

    if target_region_mask is not None:
        cats_a = cats_a * target_region_mask
        cats_b = cats_b * target_region_mask

    cats_a = F.pad(cats_a, pad_size, "constant", 0)
    cats_b = F.pad(cats_b, pad_size, "constant", 0)

    cats_a_coor = cats_a.nonzero()
    cats_b_coor = cats_b.nonzero()

    cdr_a, cdr_b = 1.0, 1.0

    if len(cats_a_coor) != 0:
        cdr_a = 0.0
        for num_edge_a, coor in enumerate(range(cats_a_coor.shape[0])):
            h, w = cats_a_coor[coor][-2], cats_a_coor[coor][-1]
            gt_sc_a = gt_a_slic[h - kernel_size:h + kernel_size, w - kernel_size:w + kernel_size] != gt_a_slic[h, w]
            inputs_sc_a = inputs_a_slic[h - kernel_size:h + kernel_size, w - kernel_size:w + kernel_size] == inputs_a_slic[h, w]

            if gt_sc_a.sum() != 0:
                cdr_a += 1 - float((gt_sc_a * inputs_sc_a).sum()) / float(gt_sc_a.sum())
            else:
                cdr_a += 1
        cdr_a /= (num_edge_a + 1)

    if len(cats_b_coor) != 0:
        cdr_b = 0.0
        for num_edge_b, coor in enumerate(range(cats_b_coor.shape[0])):
            h, w = cats_b_coor[coor][-2], cats_b_coor[coor][-1]
            gt_sc_b = gt_b_slic[h - kernel_size:h + kernel_size, w - kernel_size:w + kernel_size] != gt_b_slic[h, w]
            inputs_sc_b = inputs_b_slic[h - kernel_size:h + kernel_size, w - kernel_size:w + kernel_size] == inputs_b_slic[h, w]

            if gt_sc_b.sum() != 0:
                cdr_b += 1 - float((gt_sc_b * inputs_sc_b).sum()) / float(gt_sc_b.sum())
            else:
                cdr_b += 1
        cdr_b /= (num_edge_b + 1)

    return cdr_a, cdr_b


def compute_cdr(img_gt, img_pred, kernel_size=5, mask=None):
    lab_gt = cv2.cvtColor(img_gt, cv2.COLOR_RGB2LAB)
    lab_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2LAB)

    lab_gt = np.transpose(lab_gt, (2, 0, 1))
    lab_pred = np.transpose(lab_pred, (2, 0, 1))

    cdr_a, cdr_b = cdr(
        inputs=lab_pred,
        gt=lab_gt,
        kernel_size=kernel_size,
        target_region_mask=None,
        mask=mask
    )
    return (cdr_a + cdr_b) / 2.0


def evaluate_colorization(original_dir, colorized_dir, image_list, use_gpu=True):
    loss_fn = lpips.LPIPS(net='alex', spatial=False, version='0.1')
    if use_gpu:
        loss_fn.cuda()

    all_data = []
    for img_name in image_list:
        orig_path = os.path.join(original_dir, f'{img_name}.png')
        if type == "c":
            colorized_paths = sorted(glob(os.path.join(colorized_dir, f'{img_name}_*.png')))
        else:
            colorized_paths = sorted(glob(os.path.join(colorized_dir, f'{img_name}_{lvl}_*.png')))

        if not os.path.exists(orig_path) or not colorized_paths:
            continue

        mask = cv2.imread(f"cats/{img_name}.png", cv2.IMREAD_GRAYSCALE)
        _, mask = cv2.threshold(mask, 128, 1, cv2.THRESH_BINARY)
        mask = cv2.resize(mask, (256, 256))

        img_gt = cv2.imread(orig_path)
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gt = cv2.resize(img_gt, (256, 256))
        
        if use_gpu:
            img0 = lpips.im2tensor(lpips.load_image(orig_path)).cuda()
        else:
            lpips.im2tensor(lpips.load_image(orig_path))


        for path in colorized_paths:
            seed = os.path.basename(path).split('_')[-1].replace('.png', '')
            img_pred = cv2.imread(path)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)
            img_pred = cv2.resize(img_pred, (256, 256))
            img1 = lpips.im2tensor(lpips.load_image(path))

            if use_gpu:
                img1 = img1.cuda()

            if img0.shape[-2:] != img1.shape[-2:]:
                img1 = F.interpolate(
                    img1,
                    size=img0.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            lpips_val = loss_fn(img0, img1).item()
            delta_e = deltaE_ciede2000(rgb2lab(img_gt / 255.0), rgb2lab(img_pred / 255.0)).mean()
            cdr_val = compute_cdr(img_gt, img_pred, mask=mask)

            all_data.append({
                'Image': img_name,
                'Seed': seed,
                'LPIPS': lpips_val,
                'DeltaE': delta_e,
                'CDR': cdr_val
            })

    df = pd.DataFrame(all_data)
    df.to_csv(f'all_scores_{type}.csv', index=False)

    mean_df = df.groupby('Image').mean(numeric_only=True).reset_index()
    median_df = df.groupby('Image').median(numeric_only=True).reset_index()

    for metric in ['LPIPS', 'DeltaE', 'CDR']:
        mean_df[['Image', metric]].to_csv(f'mean_{metric.lower()}_{type}.csv', index=False)
        median_df[['Image', metric]].to_csv(f'median_{metric.lower()}_{type}.csv', index=False)
        plot_median_scores(df, median_df, metric, f'median_{metric.lower()}_{type}.png')

        if metric == 'CDR':
            best_df = df.loc[df.groupby('Image')[metric].idxmax()]
            worst_df = df.loc[df.groupby('Image')[metric].idxmin()]
        else:
            best_df = df.loc[df.groupby('Image')[metric].idxmin()]
            worst_df = df.loc[df.groupby('Image')[metric].idxmax()]

        best_df[['Image', 'Seed', metric]].to_csv(f'best_{metric.lower()}_{type}.csv', index=False)
        worst_df[['Image', 'Seed', metric]].to_csv(f'worst_{metric.lower()}_{type}.csv', index=False)

        plot_metric_summary(
            mean_df[['Image', metric]], metric,
            f'Mean {metric} per Image',
            f'mean_{metric.lower()}_{type}.png'
        )
        plot_metric_summary(
            best_df[['Image', metric]], metric,
            f'Best {metric} per Image',
            f'best_{metric.lower()}_{type}.png',
            color='seagreen'
        )
        plot_metric_summary(
            worst_df[['Image', metric]], metric,
            f'Worst {metric} per Image',
            f'worst_{metric.lower()}_{type}.png',
            color='indianred'
        )

    save_overall_metric_tables(df, type, colorized_dir, lvl, original_dir)
    return df


original_dir = './test_imgs'
image_list = ["1", "2", "3", "4", "5", "6", "7"]
type = sys.argv[1]  # adjust code based on the naming convention of the images
gt_path = sys.argv[2]
comp_path = sys.argv[3]


evaluate_colorization(gt_path, comp_path, image_list)
