from encodings import charmap
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import cv2


# ====================================================================================
# COLOR HELPERS
# ====================================================================================

def channel_rgb_tuple(ch: int):
    """
    Returns a consistent RGB color (0–1 floats) for a given channel index.

    Parameters
    ----------
    ch : int
        The index of the channel (e.g., 0, 1, 2, 3).

    Returns
    -------
    tuple(float, float, float)
        RGB color in normalized 0–1 range.

    Notes
    -----
    - Channels 0–3 get fixed blue/green/red/magenta colors.
    - Additional channels cycle through the same palette.
    - Used for composites and mask outline coloring.
    """
    if ch == 0:
        return (0.0, 0.0, 1.0)   # Blue
    if ch == 1:
        return (0.0, 1.0, 0.0)   # Green
    if ch == 2:
        return (1.0, 0.0, 0.0)   # Red
    if ch == 3:
        return (1.0, 0.0, 1.0)   # Magenta

    # fallback for higher channel numbers
    palette = [(0.0,0.0,1.0),
               (0.0,1.0,0.0),
               (1.0,0.0,0.0),
               (1.0,0.0,1.0)]
    return palette[ch % len(palette)]


# ====================================================================================
# RGB COMPOSITE PLOTTER
# ====================================================================================

def plot_channels_composite(img, channels, colors=None, title="", ax=None, use_max=False, normalisation = False):
    """
    Create an RGB composite visualization from selected channels.

    Parameters
    ----------
    img : np.ndarray
        (H, W, C) multi-channel input image from which individual channels
        will be composited.

    channels : list[int]
        Channel indices to include in the composite.

    colors : list[tuple], optional
        List of RGB triplets (0–1 floats) of the same length as `channels`.
        If None, automatically assigns colors via `channel_rgb_tuple()`.

    title : str
        Optional subplot title.

    ax : matplotlib.axes.Axes, optional
        Axes object to render into. If None, uses current axes.

    use_max : bool
        If True (recommended), channels are combined using per-pixel
        `max` to avoid channel-overbrightening.
        If False, channels are summed.

    Returns
    -------
    None
        Displays the composite image.

    Explanation
    -----------
    Each selected channel is:
        1. Extracted from img[..., ch]
        2. Normalized to 0–1 (independently)
        3. Mapped into an RGB layer using the provided or default color
        4. Merged into the composite image via per-pixel maximum

    This produces an intuitive multi-channel overlay where each channel
    has a clear color identity.
    """
    H, W, C = img.shape
    composite = np.zeros((H, W, 3), dtype=float)

    if colors is None:
        colors = [channel_rgb_tuple(ch) for ch in channels]

    for ch, col in zip(channels, colors):
        d = img[..., ch].astype(float)

            # Normalize channel to [0,1]
        if normalisation == True:
            vmax = d.max()
            if vmax > 0:
                d = d / vmax
            else:
                d = np.zeros_like(d, dtype=float)

        # Form a 3-channel colored layer
        layer = d[..., None] * np.array(col)[None, None, :]

        # Merge using max or sum
        if use_max:
            composite = np.maximum(composite, layer)
        else:
            composite += layer

    composite = np.clip(composite, 0, 1)

    if ax is None:
        ax = plt.gca()
    ax.imshow(composite)
    if title:
        ax.set_title(title)
    ax.axis('off')


# ====================================================================================
# MASK OUTLINE OVERLAY (ROBUST)
# ====================================================================================

def overlay_mask_outline(
    image,
    mask,
    color=(1,0,0),
    linewidth=2,
    alpha=0.25,
    scale='percentile'
):
    """
    Overlay the outline of a binary mask onto a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        (H, W) grayscale input image. This is rescaled for display.

    mask : np.ndarray
        (H, W) binary or thresholdable mask. Any nonzero pixel is treated
        as part of the mask.

    color : tuple(float, float, float)
        RGB color of the outline, normalized to 0–1.

    linewidth : int
        Thickness (in pixels) of the contour lines.

    alpha : float
        Blend factor for the overlay:
            - 1.0 = show only the outline version
            - 0.0 = show only original grayscale image

    scale : {'minmax', 'percentile', 'log', None}
        Method to normalize the grayscale image to [0,1] for consistent
        visualization:
            • 'minmax'     : global min–max scaling
            • 'percentile' : robust scaling using 1st–99th percentile
            • 'log'        : log compression (for high dynamic range)
            • None         : assume input already approx 0–1

    Returns
    -------
    np.ndarray
        (H, W, 3) float RGB image with outlines blended over the background.

    Explanation
    -----------
    Steps:
        1. Normalize/scale the grayscale background.
        2. Convert background to (H,W,3) RGB.
        3. Extract mask contours using OpenCV.
        4. Draw colored outlines on the background.
        5. Blend outline drawing with the grayscale image via `alpha`.

    This ensures crisp contour overlays independent of input image range.
    """
    image = np.asarray(image, dtype=np.float32)

    # ----- Scale image to [0,1] -----
    if scale == 'minmax':
        mn, mx = float(image.min()), float(image.max())
        img = (image - mn) / (mx - mn + 1e-12)
    elif scale == 'percentile':
        vmin, vmax = np.percentile(image, (1, 99))
        img = np.clip((image - vmin) / (vmax - vmin + 1e-12), 0, 1)
    elif scale == 'log':
        img = np.log1p(image)
        img = img / (img.max() + 1e-12)
    else:
        img = np.clip(image, 0, 1)

    # RGB background
    img_rgb = np.stack([img, img, img], axis=-1).astype(np.float32)

    # ----- Ensure mask is 2D -----
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {mask.shape}")

    # Convert mask for OpenCV contour extraction
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    mask_uint8 = np.ascontiguousarray(mask_uint8)

    # Detect contours
    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Prepare BGR image for OpenCV drawing
    img_uint8 = (img_rgb * 255).astype(np.uint8)
    color_bgr = (int(color[2]*255),
                 int(color[1]*255),
                 int(color[0]*255))

    # Draw outlines if any contours exist
    if len(contours) > 0:
        cv2.drawContours(img_uint8, contours, -1, color_bgr, thickness=linewidth)

    outline = img_uint8.astype(np.float32) / 255.0

    # Blend: alpha * outline + (1 - alpha) * background
    blended = alpha * outline + (1.0 - alpha) * img_rgb
    return np.clip(blended, 0, 1)


# ====================================================================================
# MASTER PLOTTING FUNCTION
# ====================================================================================


def plot_input_and_predictions_with_outline(
    idx,
    image,
    predictions,
    predictions_cleaned,
    channels=[0,1,2,3],
    mask_colors=None,
    overlay_alpha=0.25,
    linewidth=2,
    scale='percentile',
    highlight_rows=None,      # NEW: list of row indices to highlight
    highlight_colors=None     # NEW: list of colors corresponding to rows
):
    # Extract slices
    img = image[idx]
    pred = predictions[idx]
    pred_clean = predictions_cleaned[idx]

    n_channels = len(channels)
    n_cols = n_channels + 1
    n_rows = 4

    fig_w = max(12, 3 * n_cols)
    fig_h = 3.5 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_rows, n_cols + 1,
        width_ratios=[0.08] + [1.0]*n_cols,
        height_ratios=[1]*n_rows,
        wspace=0.05,
        hspace=0.22
    )

    row_labels = ["Input", "Predictions", "Cleaned\nPredictions", "Mask Outlines"]
    for r, label in enumerate(row_labels):
        ax = fig.add_subplot(gs[r, 0])
        ax.text(0.5, 0.5, label, rotation=90,
                ha='center', va='center', fontsize=12)
        ax.axis('off')

    for j, ch in enumerate(channels):
        ax = fig.add_subplot(gs[0, j+1])
        ax.set_title(f"Ch {ch}", fontsize=12)
        ax.axis('off')

    ax = fig.add_subplot(gs[0, n_cols])
    ax.set_title("Composite", fontsize=12)
    ax.axis('off')

    if mask_colors is None:
        mask_colors = [channel_rgb_tuple(ch) for ch in channels]

    def plot_comp(arr, ax):
        plot_channels_composite(
            arr,
            channels=channels,
            colors=[channel_rgb_tuple(ch) for ch in channels],
            ax=ax
        )

    arrays = [img, pred, pred_clean]

    # ----------------------------------------
    # ROWS 0–2: Input, Predictions, Cleaned
    # ----------------------------------------
    for row_idx, arr in enumerate(arrays):
        for col_idx, ch in enumerate(channels):
            ax = fig.add_subplot(gs[row_idx, col_idx+1])
            show = arr[..., ch]

            cmap = None
            is_binary = (show.dtype == np.uint8) or (np.nanmax(show) <= 1.0)
            if is_binary:
                rgb = channel_rgb_tuple(ch)
                cmap = ListedColormap(['black', rgb])

            ax.imshow(show, cmap=cmap, interpolation='nearest')
            ax.axis('off')

        # Composite image
        ax = fig.add_subplot(gs[row_idx, n_cols])
        plot_comp(arr, ax)

    # ----------------------------------------
    # ROW 3: Outline Overlays
    # ----------------------------------------
    H, W = img.shape[:2]
    composite_overlay = np.zeros((H, W, 3), dtype=np.float32)

    for col_idx, ch in enumerate(channels):
        background = img[..., ch]
        mask = pred_clean[..., ch]

        overlay_img = overlay_mask_outline(
            background,
            mask,
            color=mask_colors[col_idx % len(mask_colors)],
            linewidth=linewidth,
            alpha=overlay_alpha,
            scale=scale
        )

        ax = fig.add_subplot(gs[3, col_idx+1])
        ax.imshow(overlay_img)
        ax.axis('off')

        composite_overlay = np.maximum(composite_overlay, overlay_img)

    ax = fig.add_subplot(gs[3, n_cols])
    ax.imshow(np.clip(composite_overlay, 0, 1))
    ax.axis('off')

    # ----------------------------------------
    # Highlight multiple rows
    # ----------------------------------------
    if highlight_rows is not None:
        if highlight_colors is None:
            # Default: use red for all
            highlight_colors = ['red']*len(highlight_rows)
        pad = 0.017
        for row_idx, color in zip(highlight_rows, highlight_colors):
            row0 = gs[row_idx, 0].get_position(fig)
            row1 = gs[row_idx, n_cols].get_position(fig)
            x0 = row0.x0
            y0 = row0.y0
            x1 = row1.x1
            y1 = row1.y1
            width = x1 - x0
            height = y1 - y0
            rect = patches.Rectangle(
                (x0 - pad, y0 - pad),
                width + 2*pad,
                height + 2*pad,
                transform=fig.transFigure,
                linewidth=3, edgecolor=color, facecolor='none'
            )
            fig.patches.append(rect)

    plt.show()


