import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gabor
from skimage.feature import canny
from scipy.stats import skew, kurtosis
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --- Feature functions (same logic as your pipeline) ---

def apply_gabor_filters(image_gray, frequencies=[0.1, 0.3, 0.5], thetas=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    responses = []
    for freq in frequencies:
        row = []
        for theta in thetas:
            real, _ = gabor(image_gray, frequency=freq, theta=theta)
            m = np.mean(real)
            v = np.var(real)
            energy = np.sum(real**2)
            row.append((m, v, energy))
        responses.append(row)
    # shape: (n_freq, n_theta, 3)
    return np.array(responses)

def extract_canny_edges(image_gray, sigma=1.0):
    edges = canny(image_gray, sigma=sigma)
    return np.array([np.sum(edges)]), edges

def extract_histogram_features(image_gray, bins=10):
    hist, bin_edges = np.histogram(image_gray, bins=bins, range=(0, 255))
    return hist, bin_edges

def extract_statistical_features(image_gray):
    flat = image_gray.flatten()
    return [np.mean(flat), np.var(flat), skew(flat), kurtosis(flat)]

# --- Visualization (improved layout) ---

def visualize_image_features_fixed(image_path, output_path="/mnt/data/Figure_00_fixed.png",
                                     frequencies=None, thetas=None):
    if frequencies is None:
        frequencies = [0.1, 0.3, 0.5]
    if thetas is None:
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    # Extract features
    gabor_responses = apply_gabor_filters(image_gray, frequencies, thetas)  # (n_freq, n_theta, 3)
    canny_feat, edges = extract_canny_edges(image_gray)
    hist, bin_edges = extract_histogram_features(image_gray, bins=10)
    stats = extract_statistical_features(image_gray)  # [mean, var, skew, kurtosis]

    # prepare figure using GridSpec-like layout
    fig = plt.figure(figsize=(14, 10), constrained_layout=False)
    # Grid: top row 3 (image, edges, histogram), bottom row 3 heatmaps
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1],
                          hspace=0.35, wspace=0.35)

    # Top-left: Grayscale image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image_gray, cmap='gray', aspect='auto')
    ax_img.set_title("Grayscale Image", fontsize=12)
    ax_img.axis('off')

    # Top-middle: Canny edges
    ax_edge = fig.add_subplot(gs[0, 1])
    ax_edge.imshow(edges, cmap='gray', aspect='auto')
    ax_edge.set_title(f"Canny Edges (sum={int(canny_feat[0])})", fontsize=12)
    ax_edge.axis('off')

    # Top-right: Histogram (10 bins)
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.bar(range(len(hist)), hist, width=0.7)
    # create short readable tick labels
    tick_labels = []
    for i in range(len(bin_edges)-1):
        left = int(bin_edges[i])
        right = int(bin_edges[i+1])
        tick_labels.append(f"{left}-{right}")
    ax_hist.set_xticks(range(len(hist)))
    ax_hist.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax_hist.set_title("Pixel Intensity Histogram (10 bins)", fontsize=12)
    ax_hist.set_xlabel("Bin ranges", fontsize=10)
    ax_hist.set_ylabel("Frequency", fontsize=10)

    # Bottom row: three heatmaps (Gabor Mean, Variance, Energy)
    n_freq = gabor_responses.shape[0]
    n_theta = gabor_responses.shape[1]

    # prepare arrays for mean/var/energy
    g_mean = gabor_responses[:, :, 0]
    g_var  = gabor_responses[:, :, 1]
    g_energy = gabor_responses[:, :, 2]

    # 1st heatmap (Mean)
    ax_gm = fig.add_subplot(gs[1, 0])
    im1 = ax_gm.imshow(g_mean, aspect='auto', origin='lower')
    ax_gm.set_title("Gabor Mean (freq x theta)", fontsize=11)
    ax_gm.set_xlabel("Theta index", fontsize=9)
    ax_gm.set_ylabel("Frequency index", fontsize=9)
    ax_gm.set_xticks(range(n_theta))
    ax_gm.set_yticks(range(n_freq))
    # colorbar to the right without overlapping
    divider = make_axes_locatable(ax_gm)
    cax1 = divider.append_axes("right", size="5%", pad=0.08)
    cbar1 = fig.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=8)

    # 2nd heatmap (Variance)
    ax_gv = fig.add_subplot(gs[1, 1])
    im2 = ax_gv.imshow(g_var, aspect='auto', origin='lower')
    ax_gv.set_title("Gabor Variance (freq x theta)", fontsize=11)
    ax_gv.set_xlabel("Theta index", fontsize=9)
    ax_gv.set_ylabel("Frequency index", fontsize=9)
    ax_gv.set_xticks(range(n_theta))
    ax_gv.set_yticks(range(n_freq))
    divider = make_axes_locatable(ax_gv)
    cax2 = divider.append_axes("right", size="5%", pad=0.08)
    cbar2 = fig.colorbar(im2, cax=cax2)
    cbar2.ax.tick_params(labelsize=8)

    # 3rd heatmap (Energy)
    ax_ge = fig.add_subplot(gs[1, 2])
    im3 = ax_ge.imshow(g_energy, aspect='auto', origin='lower')
    ax_ge.set_title("Gabor Energy (freq x theta)", fontsize=11)
    ax_ge.set_xlabel("Theta index", fontsize=9)
    ax_ge.set_ylabel("Frequency index", fontsize=9)
    ax_ge.set_xticks(range(n_theta))
    ax_ge.set_yticks(range(n_freq))
    divider = make_axes_locatable(ax_ge)
    cax3 = divider.append_axes("right", size="5%", pad=0.08)
    cbar3 = fig.colorbar(im3, cax=cax3)
    cbar3.ax.tick_params(labelsize=8)

    # Stats box placed inside the energy axes but offset so it does not overlap the colorbar
    stats_text = (
        f"Mean: {stats[0]:.2f}\n"
        f"Variance: {stats[1]:.2f}\n"
        f"Skewness: {stats[2]:.2f}\n"
        f"Kurtosis: {stats[3]:.2f}"
    )
    # Use coordinates relative to ax_ge; x=0.98 would be too close to colorbar, so use 0.70
    ax_ge.text(0.70, 0.02, stats_text, transform=ax_ge.transAxes,
               fontsize=9, va='bottom', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # Final layout adjustments
    # fig.suptitle(f"Feature Visualization for: {os.path.basename(image_path)}", fontsize=14)
    # Manually adjust margins so nothing overlaps
    plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.06, hspace=0.30, wspace=0.30)

    # Save and show
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved visualization to: {output_path}")
    plt.show()


if __name__ == "__main__":
    input_path = r"C:\Users\Yuvraj\OneDrive\Files\Work\MARS\RP\3.JPG"   # change if needed
    output_path = "RP/Figure_02_fixed.png"  # change if needed
    visualize_image_features_fixed(input_path, output_path=output_path)
