import cv2
import albumentations as A
import numpy as np
# ----------------------------
# Preprocessing functions
# ----------------------------
def preprocess_fluorescence_image(image, mask, target_size=1024):
    """Normalize, resize and pad image & mask"""
    if image.ndim == 2:
        image = np.stack([image]*3, axis=-1)
    elif image.ndim == 3 and image.shape[-1] != 3:
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        elif image.shape[-1] > 3:
            image = image[..., :3]
    
    image_normalized = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[-1]):
        channel = image[..., c].astype(np.float32)
        ch_min = np.percentile(channel, 2)
        ch_max = np.percentile(channel, 98)
        image_normalized[..., c] = np.clip((channel - ch_min) / (ch_max - ch_min), 0, 1)
    
    scale = min(target_size / image.shape[0], target_size / image.shape[1])
    new_size = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    image_resized = cv2.resize(image_normalized, new_size, interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, new_size, interpolation=cv2.INTER_NEAREST)
    
    pad_h = target_size - image_resized.shape[0]
    pad_w = target_size - image_resized.shape[1]
    image_padded = np.pad(image_resized, ((0,pad_h),(0,pad_w),(0,0)), mode='constant', constant_values=0)
    mask_padded = np.pad(mask_resized, ((0,pad_h),(0,pad_w)), mode='constant', constant_values=0)
    
    return image_padded, mask_padded

# ----------------------------
# Augmentation
# ----------------------------
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_REFLECT),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.GaussNoise(p=0.3),
    A.ElasticTransform(p=0.2),
    A.Perspective(scale=(0.05, 0.1), p=0.2)
], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.1))

def augment_image_mask(image, mask):
    bboxes = [[0,0,image.shape[1],image.shape[0]]]
    augmented = augmentation_pipeline(image=image, mask=mask, bboxes=bboxes)
    return augmented['image'], augmented['mask']

# ----------------------------
# Circle-based prompts
# ----------------------------
def mask_centroid(mask):
    ys, xs = np.where(mask>0)
    if len(xs) == 0:
        return mask.shape[1]//2, mask.shape[0]//2
    return xs.mean(), ys.mean()

def circle_points(center, radius, num_points=16):
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    points = np.stack([cx + radius*np.cos(angles), cy + radius*np.sin(angles)], axis=1)
    return points

def generate_circle_prompts(mask, scale=1.2, num_points=16):
    cx, cy = mask_centroid(mask)
    r = np.sqrt(mask.sum() / np.pi) * scale
    points = circle_points((cx, cy), r, num_points)
    points = points.reshape(-1,1,2).astype(np.float32)
    labels = np.ones((points.shape[0],1), dtype=np.int32)
    return points, labels