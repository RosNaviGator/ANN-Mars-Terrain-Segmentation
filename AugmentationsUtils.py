# -------------------------------------- #
#           Import Libraries              #
# -------------------------------------- #

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import warnings
import cv2  # OpenCV for faster image processing
import logging
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter, map_coordinates
import os
import matplotlib
from matplotlib.colors import ListedColormap

# -------------------------------------- #
#          Configure Logging              #
# -------------------------------------- #

logging.basicConfig(level=logging.INFO, filename='augmentation.log',
                    format='%(asctime)s %(levelname)s:%(message)s')

# -------------------------------------- #
#       Define Label Mapping and Colors   #
# -------------------------------------- #

# Define label mapping
label_mapping = {
    0: 'Background',
    1: 'Soil',
    2: 'Bedrock',
    3: 'Sand',
    4: 'Big Rock'
}

# Define colors for each class (updated palette)
colors = [
    '#000000',  # 0: Background 
    '#A95F44',  # 1: Soil 
    '#584C44',  # 2: Bedrock 
    '#C5986F',  # 3: Sand 
    '#A69980'   # 4: Big Rock 
]

# Create a custom colormap
cmap = ListedColormap(colors)

# Function to apply label colors to the mask as RGB
def apply_label_colors(label_image):
    colored_label_image = np.zeros((label_image.shape[0], label_image.shape[1], 3), dtype=np.float32)
    for label_value, color in enumerate(colors):
        # Convert hex color to RGB normalized [0,1]
        rgb = np.array(matplotlib.colors.to_rgb(color))
        colored_label_image[label_image == label_value] = rgb
    return colored_label_image

# -------------------------------------- #
#      Define Label Distribution          #
# -------------------------------------- #

# Define label distribution (probabilities must sum to 1)
label_distribution = {
    0: 0.0,   # Background - 0%
    1: 0.20,  # Soil - 20%
    2: 0.20,  # Bedrock - 20%
    3: 0.30,  # Sand - 30%
    4: 0.30   # Big Rock - 30%
}

# -------------------------------------- #
#      Define Enabled Augmentations      #
# -------------------------------------- #

# Augmentation settings
enabled_augmentations = {
    "flip": True,
    "rotation": True,
    "zoom": True,
    "shift": True,
    "elastic": False,
    "shear": True,
    "cutout": True,
}

# -------------------------------------- #
#      Visualization Utility Functions   #
# -------------------------------------- #

def create_legend_patches():
    """
    Create legend patches for visualization.

    Returns:
        List of matplotlib.patches.Patch for the legend.
    """
    return [mpatches.Patch(color=colors[i], label=label_mapping[i]) for i in label_mapping]

def plot_images(original_image=None, original_label=None, augmented_images=None, augmented_labels=None, augmentation_labels=None, title_suffix="", fontsize=10):
    """
    Plot the original image and mask alongside each augmented version and their masks.
    If original_image and original_label are None, only augmented images and masks are plotted.

    Parameters:
        original_image (np.array, optional): The original image.
        original_label (np.array, optional): The original mask.
        augmented_images (list, optional): List of augmented images.
        augmented_labels (list, optional): List of augmented masks.
        augmentation_labels (list, optional): List of augmentation descriptions.
        title_suffix (str, optional): Suffix to add to the plot title (e.g., index information).
        fontsize (int, optional): Font size for the plot titles.
    """
    if augmented_images is None:
        augmented_images = []
    if augmented_labels is None:
        augmented_labels = []
    if augmentation_labels is None:
        augmentation_labels = []

    include_original = original_image is not None and original_label is not None
    num_aug = len(augmented_images)

    # Determine the number of columns
    if include_original:
        total_cols = num_aug + 1
    else:
        total_cols = num_aug

    # Handle case with no augmentations
    if total_cols == 0:
        print("No augmentations to display.")
        return

    # Define the layout: images on the first row, masks on the second row
    fig, axes = plt.subplots(2, total_cols, figsize=(3 * total_cols, 5))  # Standardized figsize

    # Ensure axes is 2D even if total_cols ==1
    if total_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    col_idx = 0

    # Plot original image and mask if provided
    if include_original:
        ax_img = axes[0, col_idx]
        if original_image.ndim == 2:
            ax_img.imshow(original_image, cmap='gray')
        else:
            ax_img.imshow(original_image)
        # Incorporate the title_suffix here
        ax_img.set_title(f"Original Image\n{title_suffix}", fontsize=fontsize)
        ax_img.axis('off')

        ax_mask = axes[1, col_idx]
        ax_mask.imshow(original_label, cmap=cmap, vmin=0, vmax=len(label_mapping) - 1)
        ax_mask.set_title(f"Original Mask\n{title_suffix}", fontsize=fontsize)
        ax_mask.axis('off')

        col_idx += 1

    # Plot augmented images and masks
    for aug_img, aug_lbl, aug_lbl_text in zip(augmented_images, augmented_labels, augmentation_labels):
        # Plot augmented image
        ax_img = axes[0, col_idx]
        if aug_img.ndim == 2:
            ax_img.imshow(aug_img, cmap='gray')
        else:
            ax_img.imshow(aug_img)
        ax_img.set_title(f"{aug_lbl_text}", fontsize=fontsize)
        ax_img.axis('off')

        # Plot augmented mask
        ax_mask = axes[1, col_idx]
        ax_mask.imshow(aug_lbl, cmap=cmap, vmin=0, vmax=len(label_mapping) - 1)
        ax_mask.set_title(f"{aug_lbl_text}", fontsize=fontsize)
        ax_mask.axis('off')

        col_idx += 1

    # Adjust layout and add legend
    patches = create_legend_patches()
    if include_original:
        fig.legend(handles=patches, loc='upper center', ncol=len(label_mapping), fontsize=fontsize, title="Classes")
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
    else:
        fig.legend(handles=patches, loc='upper center', ncol=len(label_mapping), fontsize=fontsize, title="Classes")
        plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.show()

def plot_augmented_pairs(original_image, original_mask,
                         source_image, source_mask,
                         augmented_image, augmented_mask,
                         indices, region_coords=None, paste_coords=None, region_size=10, fontsize=10):
    """
    Visualizes the original, source, and augmented images and masks in a single figure.

    Parameters:
        original_image (np.array): Target image before augmentation.
        original_mask (np.array): Mask of the target image before augmentation.
        source_image (np.array): Source image from which the region is copied.
        source_mask (np.array): Mask of the source image.
        augmented_image (np.array): Target image after pasting the region.
        augmented_mask (np.array): Mask of the target image after pasting.
        indices (tuple): Indices of target and source images.
        region_coords (tuple, optional): (top, left) of the copied region in the source mask.
        paste_coords (tuple, optional): (top, left) where the region was pasted in the target mask.
        region_size (int): Size of the copied region.
        fontsize (int, optional): Font size for the plot titles.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Standardized figsize

    titles = [
        f"Original Image\n(Index: {indices[0]})",
        f"Source Image\n(Index: {indices[1]})",
        "Augmented Image",
        f"Original Mask\n(Index: {indices[0]})",
        f"Source Mask\n(Index: {indices[1]})",
        "Augmented Mask"
    ]

    # List of images and their corresponding display parameters
    images = [
        (original_image, 'gray', None),
        (source_image, 'gray', region_coords),
        (augmented_image, 'gray', paste_coords),
        (original_mask, cmap, None),
        (source_mask, cmap, region_coords),
        (augmented_mask, cmap, paste_coords)
    ]

    for ax, (img, cmap_used, coords), title in zip(axes.flatten(), images, titles):
        if cmap_used == 'gray':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=len(label_mapping)-1)

        ax.set_title(title, fontsize=fontsize)
        ax.axis("off")

        # Add highlight rectangle if coordinates are provided
        if coords:
            top, left = coords
            rect = mpatches.FancyBboxPatch(
                (left, top),
                region_size,
                region_size,
                boxstyle="round,pad=0.02",
                linewidth=2,
                edgecolor='white',
                facecolor='none'
            )
            ax.add_patch(rect)

    # Create a legend for the mask classes
    # Place legend below the subplots
    patches = [mpatches.Patch(color=colors[i], label=label_mapping[i]) for i in label_mapping]
    fig.legend(handles=patches, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=len(label_mapping), fontsize=fontsize, title="Classes")

    plt.tight_layout()
    plt.show()

def copy_paste_images_and_plot(target_idx, source_idx, X, y, region_size=10):
    """
    Applies copy-and-paste augmentation between two specified images and visualizes the result.

    Parameters:
        target_idx (int): Index of the target image.
        source_idx (int): Index of the source image.
        X (np.array): Dataset images.
        y (np.array): Dataset labels.
        region_size (int): Size of the square region to copy and paste.
    """
    original_image = X[target_idx].squeeze()
    original_mask = y[target_idx].squeeze()
    source_image = X[source_idx].squeeze()
    source_mask = y[source_idx].squeeze()

    # Select a random non-background label
    non_background_labels = [label for label in label_mapping if label != 0]
    selected_label = random.choice(non_background_labels)
    region_coords = extract_label_region(source_mask, selected_label, region_size=region_size)

    if region_coords is None:
        logging.error(f"No suitable region found in source image {source_idx} for label {selected_label}.")
        print(f"No suitable region found in source image {source_idx} for label {selected_label}.")
        return

    src_top, src_left = region_coords
    region_image = source_image[src_top:src_top + region_size, src_left:src_left + region_size].copy()
    region_label = source_mask[src_top:src_top + region_size, src_left:src_left + region_size].copy()

    # Random paste location in target image
    h, w = original_mask.shape
    if h < region_size or w < region_size:
        logging.error(f"Target image {target_idx} is smaller than the region size.")
        print(f"Target image {target_idx} is smaller than the region size.")
        return

    paste_top = random.randint(0, h - region_size)
    paste_left = random.randint(0, w - region_size)

    # Perform copy-and-paste
    augmented_image = original_image.copy()
    augmented_mask = original_mask.copy()
    augmented_image[paste_top:paste_top + region_size, paste_left:paste_left + region_size] = region_image
    augmented_mask[paste_top:paste_top + region_size, paste_left:paste_left + region_size] = region_label

    # Visualize the augmentation with reduced font size
    plot_augmented_pairs(
        original_image=original_image,
        original_mask=original_mask,
        source_image=source_image,
        source_mask=source_mask,
        augmented_image=augmented_image,
        augmented_mask=augmented_mask,
        indices=(target_idx, source_idx),
        region_coords=(src_top, src_left),
        paste_coords=(paste_top, paste_left),
        region_size=region_size,
        fontsize=8  # Reduced font size
    )

# -------------------------------------- #
#      Define Augmentation Functions     #
# -------------------------------------- #

# ### 1. Elastic Deformation

def elastic_deformation(image, label, alpha=34, sigma=4):
    """
    Apply elastic deformation to image and label.

    Parameters:
        image (np.array): Grayscale image.
        label (np.array): Corresponding label mask.
        alpha (float): Scaling factor for deformation.
        sigma (float): Smoothing factor for Gaussian filter.

    Returns:
        Tuple of augmented image, augmented label, and augmentation label.
    """
    random_state = np.random.RandomState(None)
    shape = image.shape[:2]

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="reflect") * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = (y + dy).reshape(-1), (x + dx).reshape(-1)

    distorted_image = map_coordinates(image, indices, order=1).reshape(shape)
    distorted_label = map_coordinates(label, indices, order=0).reshape(shape)

    return distorted_image, distorted_label, 'Elastic Deformation'

# ### 2. Shearing

def random_shear(image, label, shear_range=10):
    """
    Apply random shear transformation to image and label.

    Parameters:
        image (np.array): Grayscale image.
        label (np.array): Corresponding label mask.
        shear_range (float): Maximum shear angle in degrees.

    Returns:
        Tuple of sheared image, sheared label, and augmentation label.
    """
    shear = random.uniform(-shear_range, shear_range)
    M = np.float32([[1, np.tan(np.radians(shear)), 0],
                    [0, 1, 0]])
    h, w = image.shape[:2]
    sheared_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    sheared_label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return sheared_image, sheared_label, f'Shear ({shear:.2f}°)'

# ### 3. Cutout

def cutout(image, label, mask_size=16):
    """
    Apply Cutout augmentation by masking out a square region in the image and label.

    Parameters:
        image (np.array): Grayscale image.
        label (np.array): Corresponding label mask.
        mask_size (int): Size of the square mask.

    Returns:
        Tuple of cutout image, cutout label, and augmentation label.
    """
    h, w = image.shape
    top = random.randint(0, h - mask_size)
    left = random.randint(0, w - mask_size)

    image_cutout = image.copy()
    label_cutout = label.copy()

    image_cutout[top:top + mask_size, left:left + mask_size] = 0  # Assuming 0 is the background
    label_cutout[top:top + mask_size, left:left + mask_size] = 0

    return image_cutout, label_cutout, f'Cutout (size={mask_size})'

# ### 4. Existing Augmentation Functions

def random_flip(image, label):
    augmentations_applied = []
    if random.random() > 0.5:
        image = np.flip(image, axis=1)  # Horizontal flip
        label = np.flip(label, axis=1)
        augmentations_applied.append('Horizontal Flip')
    if random.random() > 0.5:
        image = np.flip(image, axis=0)  # Vertical flip
        label = np.flip(label, axis=0)
        augmentations_applied.append('Vertical Flip')
    return image, label, augmentations_applied

def random_rotation(image, label):
    angle = random.choice([90, 180, 270])
    k = angle // 90
    image = np.rot90(image, k=k)
    label = np.rot90(label, k=k)
    # Ensure the image is resized back to the original 64x128 shape
    image = cv2.resize(image, (128, 64), interpolation=cv2.INTER_LINEAR)
    label = cv2.resize(label, (128, 64), interpolation=cv2.INTER_NEAREST)  # Use NEAREST for labels
    return image, label, f'Rotation {angle}°'

def random_zoom(image, label, zoom_range=(0.6, 1.4)):
    """
    Apply random zoom to image and label.

    Parameters:
        image (np.array): Grayscale image.
        label (np.array): Corresponding label mask.
        zoom_range (tuple): Range of zoom factors.

    Returns:
        Tuple of zoomed image, zoomed label, and augmentation label.
    """
    zoom_factor = random.uniform(*zoom_range)
    h, w = image.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

    # Resize image and label
    zoomed_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    zoomed_label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Use NEAREST for labels

    if zoom_factor > 1.0:  # Crop
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        zoomed_image = zoomed_image[start_h:start_h + h, start_w:start_w + w]
        zoomed_label = zoomed_label[start_h:start_h + h, start_w:start_w + w]
        augmentation = f'Zoom In ({zoom_factor:.2f}x)'
    else:  # Pad
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        # Ensure padding is non-negative
        pad_h = max(pad_h, 0)
        pad_w = max(pad_w, 0)
        # Pad image
        zoomed_image = np.pad(
            zoomed_image,
            ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)),
            mode='constant',
            constant_values=0
        )
        # Pad label
        zoomed_label = np.pad(
            zoomed_label,
            ((pad_h, h - new_h - pad_h), (pad_w, w - new_w - pad_w)),
            mode='constant',
            constant_values=0
        )
        augmentation = f'Zoom Out ({zoom_factor:.2f}x)'

    # Resize to original size (64x128) if necessary
    zoomed_image = cv2.resize(zoomed_image, (128, 64), interpolation=cv2.INTER_LINEAR)
    zoomed_label = cv2.resize(zoomed_label, (128, 64), interpolation=cv2.INTER_NEAREST)  # Use NEAREST for labels

    return zoomed_image, zoomed_label, augmentation

def random_shift(image, label, shift_range=(0.2, 0.2)):
    h, w = image.shape[:2]
    max_dx = int(shift_range[0] * w)
    max_dy = int(shift_range[1] * h)
    dx = random.randint(-max_dx, max_dx)
    dy = random.randint(-max_dy, max_dy)

    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_image = cv2.warpAffine(
        image, M, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    shifted_label = cv2.warpAffine(
        label, M, (w, h), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # Resize back to original size (64x128) if necessary
    shifted_image = cv2.resize(shifted_image, (128, 64), interpolation=cv2.INTER_LINEAR)
    shifted_label = cv2.resize(shifted_label, (128, 64), interpolation=cv2.INTER_NEAREST)  # Use NEAREST for labels

    augmentation = f'Shift (dx={dx}, dy={dy})'
    return shifted_image, shifted_label, augmentation

# -------------------------------------- #
#      Define Augmentation Pipeline      #
# -------------------------------------- #

def apply_augmentations_pipeline(indices, augmentations, X, y, copy_paste=False, region_size=10, fontsize=8):
    """
    Apply a pipeline of augmentations to a list of indices and plot the outcomes.
    Applies all specified augmentations sequentially first, then applies copy-paste if enabled.

    Parameters:
        indices (list): List of integer indices.
        augmentations (list): List of augmentation function names as strings.
        X (np.array): Dataset images with shape (num_samples, height, width, channels).
        y (np.array): Dataset labels with shape (num_samples, height, width, 1).
        copy_paste (bool): Whether to perform copy-paste after augmentations.
        region_size (int): Size of the square region to copy and paste if copy_paste is True.
        fontsize (int): Font size for the plot titles.
    """
    # Map augmentation names to functions
    augmentation_functions = {
        "flip": random_flip,
        "rotation": random_rotation,
        "zoom": random_zoom,
        "shift": random_shift,
        "elastic": elastic_deformation,
        "shear": random_shear,
        "cutout": cutout
    }

    # Analyze label distribution to get label_to_indices mapping
    label_to_indices = analyze_label_distribution(y, region_size=region_size)

    for idx in indices:
        original_image = X[idx].squeeze()
        original_mask = y[idx].squeeze()

        # Initialize working image and mask
        working_image = original_image.copy()
        working_mask = original_mask.copy()

        augmented_images = []
        augmented_labels = []
        augmentation_labels = []

        # Apply all specified augmentations sequentially
        for aug in augmentations:
            if aug in augmentation_functions:
                aug_func = augmentation_functions[aug]
                # Apply augmentation
                aug_img, aug_lbl, aug_lbl_text = aug_func(working_image, working_mask)
                # Update working image and mask
                working_image, working_mask = aug_img, aug_lbl
                # Append the augmented image and mask
                augmented_images.append(aug_img)
                augmented_labels.append(aug_lbl)
                augmentation_labels.append(aug_lbl_text)
            else:
                logging.warning(f"Augmentation '{aug}' is not recognized and will be skipped.")

        # Apply copy-paste augmentation if enabled
        if copy_paste:
            # Select a label randomly (excluding background)
            available_labels = [label for label in label_mapping.keys() if label != 0]
            if not available_labels:
                logging.warning("No labels available for copy-paste augmentation.")
            else:
                selected_label = random.choice(available_labels)

                # Check if there are source images for the selected label
                if selected_label in label_to_indices and len(label_to_indices[selected_label]) > 0:
                    # Choose a source image index randomly
                    source_idx = random.choice(label_to_indices[selected_label])

                    source_image = X[source_idx].squeeze()
                    source_mask = y[source_idx].squeeze()

                    # Extract region from source image
                    region_coords = extract_label_region(source_mask, selected_label, region_size=region_size)

                    if region_coords is not None:
                        src_top, src_left = region_coords
                        region_image = source_image[src_top:src_top + region_size, src_left:src_left + region_size].copy()
                        region_label = source_mask[src_top:src_top + region_size, src_left:src_left + region_size].copy()

                        # Define paste location randomly in the target image
                        h, w = working_mask.shape
                        if h >= region_size and w >= region_size:
                            paste_top = random.randint(0, h - region_size)
                            paste_left = random.randint(0, w - region_size)

                            # Create copy-paste augmented image
                            augmented_image_cp = working_image.copy()
                            augmented_mask_cp = working_mask.copy()

                            # Paste the region into the target image and label
                            augmented_image_cp[paste_top:paste_top + region_size, paste_left:paste_left + region_size] = region_image
                            augmented_mask_cp[paste_top:paste_top + region_size, paste_left:paste_left + region_size] = region_label

                            # Append to augmented lists
                            augmented_images.append(augmented_image_cp)
                            augmented_labels.append(augmented_mask_cp)
                            augmentation_labels.append(f'Copy-Paste from {source_idx}')
                        else:
                            logging.warning(f"Image {idx} is smaller than the region size for copy-paste.")
                    else:
                        logging.warning(f"No suitable region found in source image {source_idx} for label {selected_label}.")
                else:
                    logging.warning(f"No source images available for label '{label_mapping[selected_label]}' ({selected_label}). Copy-paste not applied for image {idx}.")

        # Plot the augmented images and masks, including the original image and its index
        plot_images(original_image=original_image, original_label=original_mask,
                   augmented_images=augmented_images,
                   augmented_labels=augmented_labels,
                   augmentation_labels=augmentation_labels,
                   title_suffix=f" (Index: {idx})",
                   fontsize=fontsize)  # Pass reduced fontsize

# -------------------------------------- #
#      Define Utility Functions          #
# -------------------------------------- #

def has_sufficient_label_region(label_mask, target_label, region_size=10):
    """
    Efficiently checks if the label mask contains at least one square region of the target label
    with the specified region size using convolution.

    Parameters:
        label_mask (np.array): Label mask with shape (height, width).
        target_label (int): The label to check for.
        region_size (int): The size of the square region to search for.

    Returns:
        bool: True if at least one such region exists, False otherwise.
    """
    binary_mask = (label_mask == target_label).astype(np.uint8)
    kernel = np.ones((region_size, region_size), dtype=np.uint8)
    conv_result = convolve2d(binary_mask, kernel, mode='valid')
    if np.any(conv_result == region_size * region_size):
        return True
    return False

def extract_label_region(label_mask, target_label, region_size=10):
    """
    Efficiently extracts a random square region of the target label from the label mask
    using convolution.

    Parameters:
        label_mask (np.array): Label mask with shape (height, width).
        target_label (int): The label to extract.
        region_size (int): The size of the square region to extract.

    Returns:
        Tuple: (top, left) coordinates if a region is found, else None.
    """
    binary_mask = (label_mask == target_label).astype(np.uint8)
    kernel = np.ones((region_size, region_size), dtype=np.uint8)
    conv_result = convolve2d(binary_mask, kernel, mode='valid')
    valid_positions = np.argwhere(conv_result == region_size * region_size)
    if valid_positions.size == 0:
        return None
    chosen_position = valid_positions[random.randint(0, len(valid_positions) - 1)]
    top, left = chosen_position
    return top, left

def analyze_label_distribution(y, region_size=10):
    """
    Analyze the label distribution in each image based on the presence of at least one
    square region of the target label with the specified region size using vectorized operations.

    Parameters:
        y (np.array): Label masks with shape (num_samples, height, width, 1).
        region_size (int): The size of the square region to search for.

    Returns:
        dict: A dictionary mapping each label to a list of image indices containing that label.
    """
    label_to_indices = defaultdict(list)
    for idx in range(y.shape[0]):
        label_mask = y[idx, :, :, 0]
        for label in label_mapping:
            if label == 0:
                continue  # Skip background
            if has_sufficient_label_region(label_mask, label, region_size=region_size):
                label_to_indices[label].append(idx)
    return label_to_indices

# -------------------------------------- #
#      Define Copy and Paste Augmentation #
# -------------------------------------- #

def apply_copy_and_paste_grouped(X, y, label_distribution,
                                 region_size=10, num_plots=5):
    """
    Apply the copy-and-paste augmentation across the dataset based on label distribution.

    Parameters:
        X (np.array): Dataset images with shape (num_samples, height, width, channels).
        y (np.array): Dataset labels with shape (num_samples, height, width, 1).
        label_distribution (dict): Dictionary mapping labels to their desired augmentation probabilities.
        region_size (int): Size of the square region to copy and paste.
        num_plots (int): Number of augmented image pairs to visualize.

    Returns:
        Tuple of augmented X and y datasets as numpy arrays.
    """
    augmented_X = []
    augmented_y = []
    plot_count = 0

    # Analyze label distribution to identify available source images
    label_to_indices = analyze_label_distribution(y, region_size=region_size)

    # Filter labels with available source images
    available_labels = [label for label in label_to_indices if label_to_indices[label]]
    if not available_labels:
        logging.error("No available labels with sufficient regions for augmentation.")
        return None

    # Prepare label selection based on distribution
    # Exclude labels with 0 probability
    labels = []
    probabilities = []
    for label, prob in label_distribution.items():
        if label == 0:
            continue  # Exclude background
        if label in available_labels:
            labels.append(label)
            probabilities.append(prob)
    total_prob = sum(probabilities)
    if total_prob == 0:
        logging.error("Total probability for label_distribution is zero after excluding unavailable labels.")
        return None
    probabilities = [p / total_prob for p in probabilities]

    # Shuffle source pools for randomness
    for label in labels:
        random.shuffle(label_to_indices[label])

    # Shuffle target image pool
    target_pool = list(range(X.shape[0]))
    random.shuffle(target_pool)

    logging.info("Starting copy-and-paste augmentation process...")

    # Perform augmentation
    for tgt_idx in target_pool:
        try:
            # Select label based on distribution
            selected_label = random.choices(labels, weights=probabilities, k=1)[0]
            logging.info(f"Selected label for augmentation: {label_mapping[selected_label]} ({selected_label})")

            # Ensure source images are available for the selected label
            if not label_to_indices[selected_label]:
                logging.warning(f"No source images available for label '{label_mapping[selected_label]}' ({selected_label}).")
                continue

            # Select a random source image
            src_idx = random.choice(label_to_indices[selected_label])
            logging.info(f"Selected source image index: {src_idx}")

            # Extract region coordinates from source mask
            label_mask = y[src_idx].squeeze()
            region_coords = extract_label_region(label_mask, selected_label, region_size=region_size)
            if region_coords is None:
                logging.warning(f"No suitable {region_size}x{region_size} region found in source image {src_idx}.")
                continue
            src_top, src_left = region_coords
            logging.info(f"Extracted region from source image {src_idx}: Top={src_top}, Left={src_left}")

            # Extract region from source image and mask
            source_image = X[src_idx].squeeze()
            source_label = y[src_idx].squeeze()[src_top:src_top + region_size, src_left:src_left + region_size].copy()
            region_image = source_image[src_top:src_top + region_size, src_left:src_left + region_size].copy()

            # Select random paste location in target image
            target_image = X[tgt_idx].squeeze()
            target_label = y[tgt_idx].squeeze()
            h, w = target_label.shape
            if h < region_size or w < region_size:
                logging.warning(f"Target image {tgt_idx} is smaller than region size. Skipping.")
                continue
            paste_top = random.randint(0, h - region_size)
            paste_left = random.randint(0, w - region_size)
            logging.info(f"Pasting region at Top={paste_top}, Left={paste_left} in target image {tgt_idx}")

            # Perform copy-and-paste
            augmented_image = target_image.copy()
            augmented_mask = target_label.copy()
            augmented_image[paste_top:paste_top + region_size, paste_left:paste_left + region_size] = region_image
            augmented_mask[paste_top:paste_top + region_size, paste_left:paste_left + region_size] = source_label

            # Append augmented data
            augmented_X.append(augmented_image[..., np.newaxis])
            augmented_y.append(augmented_mask[..., np.newaxis])

            # Plot first `num_plots` augmentations with reduced fontsize
            if plot_count < num_plots:
                plot_augmented_pairs(
                    original_image=target_image,
                    original_mask=target_label,
                    source_image=source_image,
                    source_mask=source_mask,
                    augmented_image=augmented_image,
                    augmented_mask=augmented_mask,
                    indices=(tgt_idx, src_idx),
                    region_coords=(src_top, src_left),
                    paste_coords=(paste_top, paste_left),
                    region_size=region_size,
                    fontsize=8  # Reduced font size
                )
                plot_count += 1

        except Exception as e:
            logging.error(f"Error during augmentation: {e}")
            continue

    # Convert to numpy arrays
    augmented_X = np.array(augmented_X) if augmented_X else np.empty((0, X.shape[1], X.shape[2], X.shape[3]))
    augmented_y = np.array(augmented_y) if augmented_y else np.empty((0, y.shape[1], y.shape[2], y.shape[3]))

    logging.info(f"Total augmented images: {len(augmented_X)}")
    return augmented_X, augmented_y

# -------------------------------------- #
#      Define New Augmentation Functions  #
# -------------------------------------- #

def plot_selected_images_and_masks(indices, X, y):
    """
    Plot all selected images in a single row and their corresponding masks in the row below.

    Parameters:
        indices (list): List of integer indices.
        X (np.array): Dataset images with shape (num_samples, height, width, channels).
        y (np.array): Dataset labels with shape (num_samples, height, width, 1).
    """
    num_images = len(indices)
    fig, axes = plt.subplots(2, num_images, figsize=(5 * num_images, 10))  # Standardized figsize

    # Ensure axes is 2D even if num_images ==1
    if num_images == 1:
        axes = np.expand_dims(axes, axis=1)

    for idx, img_idx in enumerate(indices):
        original_image = X[img_idx].squeeze()
        original_mask = y[img_idx].squeeze()

        # Plot image
        ax_img = axes[0, idx]
        if original_image.ndim == 2:
            ax_img.imshow(original_image, cmap='gray')
        else:
            ax_img.imshow(original_image)
        ax_img.set_title(f"Image {img_idx}", fontsize=12)
        ax_img.axis('off')

        # Plot mask
        ax_mask = axes[1, idx]
        ax_mask.imshow(original_mask, cmap=cmap, vmin=0, vmax=len(label_mapping) - 1)
        ax_mask.set_title(f"Mask {img_idx}", fontsize=12)
        ax_mask.axis('off')

    # Create and add legend to the figure
    patches = create_legend_patches()
    fig.legend(handles=patches, loc='upper center', ncol=len(label_mapping), fontsize=10, title="Classes")

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
    plt.show()

def plot_single_augmentations(indices, augmentations, X, y):
    """
    Apply single augmentations to selected images and plot the original and augmented images with masks.

    Parameters:
        indices (list): List of image indices to augment.
        augmentations (list): List of augmentation function names as strings.
        X (np.array): Dataset images with shape (num_samples, height, width, channels).
        y (np.array): Dataset labels with shape (num_samples, height, width, 1).
    """
    # Map augmentation names to functions
    augmentation_functions = {
        "flip": random_flip,
        "rotation": random_rotation,
        "zoom": random_zoom,
        "shift": random_shift,
        "elastic": elastic_deformation,
        "shear": random_shear,
        "cutout": cutout
    }

    for idx in indices:
        original_image = X[idx].squeeze()
        original_mask = y[idx].squeeze()

        # Number of augmentations to apply
        num_augs = len(augmentations)

        # Create subplots: one row for images, one for masks
        fig, axes = plt.subplots(2, num_augs + 1, figsize=(5 * (num_augs + 1), 10))

        # Handle the case when there is only one augmentation
        if num_augs == 0:
            print("No augmentations specified.")
            plt.close(fig)
            continue

        # Plot original image
        ax = axes[0, 0]
        if original_image.ndim == 2:
            ax.imshow(original_image, cmap='gray')
        else:
            ax.imshow(original_image)
        ax.set_title(f"Original Image\n(Index: {idx})", fontsize=12)
        ax.axis('off')

        # Plot original mask
        ax = axes[1, 0]
        ax.imshow(original_mask, cmap=cmap, vmin=0, vmax=len(label_mapping) - 1)
        ax.set_title(f"Original Mask\n(Index: {idx})", fontsize=12)
        ax.axis('off')

        # Apply each augmentation and plot
        for i, aug in enumerate(augmentations):
            if aug in augmentation_functions:
                aug_func = augmentation_functions[aug]
                # Apply augmentation
                aug_img, aug_lbl, aug_lbl_text = aug_func(original_image.copy(), original_mask.copy())

                # Plot augmented image
                ax_img_aug = axes[0, i + 1]
                if aug_img.ndim == 2:
                    ax_img_aug.imshow(aug_img, cmap='gray')
                else:
                    ax_img_aug.imshow(aug_img)
                ax_img_aug.set_title(f"{aug_lbl_text}", fontsize=12)
                ax_img_aug.axis('off')

                # Plot augmented mask
                ax_mask_aug = axes[1, i + 1]
                ax_mask_aug.imshow(aug_lbl, cmap=cmap, vmin=0, vmax=len(label_mapping) - 1)
                ax_mask_aug.set_title(f"{aug_lbl_text}", fontsize=12)
                ax_mask_aug.axis('off')
            else:
                logging.warning(f"Augmentation '{aug}' is not recognized and will be skipped.")
                # Plot empty subplot with warning
                ax_img_aug = axes[0, i + 1]
                ax_img_aug.text(0.5, 0.5, "Augmentation Not Found", horizontalalignment='center',
                                verticalalignment='center', fontsize=12, color='red')
                ax_img_aug.axis('off')

                ax_mask_aug = axes[1, i + 1]
                ax_mask_aug.text(0.5, 0.5, "Augmentation Not Found", horizontalalignment='center',
                                 verticalalignment='center', fontsize=12, color='red')
                ax_mask_aug.axis('off')

        # Create and add legend to the figure
        patches = create_legend_patches()
        fig.legend(handles=patches, loc='upper center', ncol=len(label_mapping), fontsize=10, title="Classes")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for the legend
        plt.show()

# -------------------------------------- #
#      Define Augmentation Pipeline      #
# -------------------------------------- #

def apply_random_augmentation(image, label):
    """
    Apply a random set of augmentations (2 to 4) to the image and label.

    Parameters:
        image (np.array): Grayscale image.
        label (np.array): Corresponding label mask.

    Returns:
        Tuple of augmented image, augmented label, and augmentation label.
    """
    augmentations = []
    augmentation_functions = []

    # Gather enabled augmentations
    for aug in enabled_augmentations:
        if enabled_augmentations[aug]:
            augmentation_functions.append(aug)

    if not augmentation_functions:
        return image, label, "No Augmentation"

    # Determine number of augmentations to apply
    num_to_apply = random.randint(2, 4)
    selected_augmentations = random.sample(augmentation_functions, min(num_to_apply, len(augmentation_functions)))

    aug_image, aug_label = image, label
    for aug in selected_augmentations:
        if aug == "flip":
            aug_image, aug_label, applied = random_flip(aug_image, aug_label)
            if applied:
                augmentations.extend(applied)
            else:
                augmentations.append("Flip (No Change)")
        elif aug == "rotation":
            aug_image, aug_label, applied = random_rotation(aug_image, aug_label)
            augmentations.append(applied)
        elif aug == "zoom":
            aug_image, aug_label, applied = random_zoom(aug_image, aug_label)
            augmentations.append(applied)
        elif aug == "shift":
            aug_image, aug_label, applied = random_shift(aug_image, aug_label)
            augmentations.append(applied)
        elif aug == "elastic":
            aug_image, aug_label, applied = elastic_deformation(aug_image, aug_label)
            augmentations.append(applied)
        elif aug == "shear":
            aug_image, aug_label, applied = random_shear(aug_image, aug_label)
            augmentations.append(applied)
        elif aug == "cutout":
            aug_image, aug_label, applied = cutout(aug_image, aug_label)
            augmentations.append(applied)

    augmentation_label = ', '.join(augmentations) if augmentations else "No Augmentation"
    return aug_image, aug_label, augmentation_label

def apply_augmentations(X, y, num_augmented=4, plot_index=None):
    """
    Apply augmentations to the entire dataset to produce a specified number of augmented images per original.

    Parameters:
        X (np.array): Dataset images with shape (num_samples, height, width, 1).
        y (np.array): Dataset labels with shape (num_samples, height, width, 1).
        num_augmented (int): Number of augmented images to create per original image.
        plot_index (int): Index of the image to visualize augmentations for.

    Returns:
        Tuple of augmented X and y datasets.
    """
    augmented_X = []
    augmented_y = []
    augmentation_labels = []

    for i in tqdm(range(X.shape[0]), desc="Augmenting data"):
        image, label = X[i].copy(), y[i].copy()

        # Ensure the original image and label have shape (64, 128, 1)
        if image.ndim == 2:  # If image has shape (64, 128), add channel dimension
            image = np.expand_dims(image, axis=-1)  # Convert to (64, 128, 1)
        if label.ndim == 2:  # If label has shape (64, 128), add channel dimension
            label = np.expand_dims(label, axis=-1)  # Convert to (64, 128, 1)

        # Add the original image and label
        augmented_X.append(image.squeeze())
        augmented_y.append(label.squeeze())
        augmentation_labels.append("Original")

        # Apply augmentations
        for _ in range(num_augmented):
            aug_image, aug_label, aug_label_text = apply_random_augmentation(
                image.squeeze(),
                label.squeeze()
            )
            augmented_X.append(aug_image)
            augmented_y.append(aug_label)
            augmentation_labels.append(aug_label_text)

        # If plotting is requested for this index, plot and continue
        if plot_index is not None and i == plot_index:
            # Select the augmented versions for plotting (excluding the original)
            start_idx = i * (num_augmented + 1) + 1
            end_idx = start_idx + num_augmented
            plot_augmented_images = augmented_X[start_idx:end_idx]
            plot_augmented_labels = augmented_y[start_idx:end_idx]
            plot_aug_labels_text = augmentation_labels[start_idx:end_idx]

            plot_images(
                image.squeeze(),
                label.squeeze(),
                plot_augmented_images,
                plot_augmented_labels,
                plot_aug_labels_text,
                fontsize=8  # Reduced font size
            )

    # Convert lists to numpy arrays and add channel dimension
    augmented_X = np.expand_dims(np.array(augmented_X), axis=-1)
    augmented_y = np.expand_dims(np.array(augmented_y), axis=-1)

    return augmented_X, augmented_y 

# -------------------------------------- #
#           Preprocess Data              #
# -------------------------------------- #

def preprocess_data(datapath):
    """
    Preprocesses the dataset located at the given datapath.

    Parameters:
    - datapath (str): Path to the .npz file containing the dataset.

    Returns:
    - X_train (np.ndarray): Preprocessed training images.
    - y_train (np.ndarray): Preprocessed training masks.
    """
    # Load the data from the .npz file
    try:
        data = np.load(datapath, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load data from {datapath}: {e}")

    # Inspect available keys in the .npz file
    available_keys = list(data.keys())
    print(f"Available keys in the .npz file: {available_keys}")

    # Ensure 'training_set' exists
    if "training_set" not in data:
        raise KeyError("The provided data does not contain a 'training_set' key.")

    training_set = data["training_set"]

    # Verify the structure of 'training_set'
    if not isinstance(training_set, (np.ndarray, list)):
        raise TypeError(f"'training_set' should be a NumPy array or list, got {type(training_set)} instead.")

    # Stack images and masks into separate NumPy arrays
    try:
        # If 'training_set' is a list of tuples/lists
        if isinstance(training_set, list):
            training_set = np.array(training_set)
        
        X_train = np.stack(training_set[:, 0], axis=0)  # Images
        y_train = np.stack(training_set[:, 1], axis=0)  # Masks
    except Exception as e:
        raise ValueError(f"Error stacking training data: {e}")

    # Preprocess images
    if X_train.ndim == 3:
        # Add a channel dimension if missing (e.g., grayscale images)
        X_train = X_train[..., np.newaxis]

    # Normalize images to [0, 1]
    X_train = X_train.astype(np.float32) / 255.0

    # Preprocess masks
    if y_train.ndim == 3:
        # Add a channel dimension if missing
        y_train = y_train[..., np.newaxis]

    # Extract the dataset name from the file name
    data_name = os.path.splitext(os.path.basename(datapath))[0]

    # Verify the number of samples
    original_count = X_train.shape[0]
    print(f"Number of images in the '{data_name}' dataset: {original_count}")

    return X_train, y_train
