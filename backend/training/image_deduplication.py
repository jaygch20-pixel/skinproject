"""Shared exact-image similarity helpers for the dataset pipeline."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np


PHASH_HAMMING_THRESHOLD = 4
SSIM_THRESHOLD = 0.98


def _perceptual_hash(image_path: str) -> int:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to read image for near-duplicate detection: {image_path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(resized)[:8, :8]
    median = np.median(low_frequency[1:, 1:])
    result = 0
    for bit in (low_frequency > median).flat:
        result = (result << 1) | int(bit)
    return result


def _structural_similarity(first_path: str, second_path: str) -> float:
    first = cv2.imread(first_path, cv2.IMREAD_GRAYSCALE)
    second = cv2.imread(second_path, cv2.IMREAD_GRAYSCALE)
    if first is None or second is None:
        raise ValueError("Unable to read an image for near-duplicate verification.")

    first = cv2.resize(first, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    second = cv2.resize(second, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mean_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mean_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    variance_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mean_first * mean_first
    variance_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mean_second * mean_second
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mean_first * mean_second
    numerator = (2 * mean_first * mean_second + c1) * (2 * covariance + c2)
    denominator = (mean_first * mean_first + mean_second * mean_second + c1) * (variance_first + variance_second + c2)
    return float(np.mean(numerator / denominator))


def detect_near_duplicate_pairs(image_paths: Iterable[str]) -> List[Dict[str, object]]:
    """Return high-confidence near-duplicate pairs among images, in stable path order."""
    buckets = defaultdict(list)
    indexed_images = []
    duplicate_pairs = []

    for image_path in sorted(str(Path(path)) for path in image_paths):
        image_hash = _perceptual_hash(image_path)
        candidates = set()
        for chunk in range(5):
            key = (chunk, (image_hash >> (chunk * 13)) & 0x1FFF)
            candidates.update(buckets[key])

        for prior_index in sorted(candidates):
            prior_path, prior_hash = indexed_images[prior_index]
            distance = (image_hash ^ prior_hash).bit_count()
            if distance <= PHASH_HAMMING_THRESHOLD:
                similarity = _structural_similarity(image_path, prior_path)
                if similarity >= SSIM_THRESHOLD:
                    duplicate_pairs.append({
                        "representative": prior_path,
                        "duplicate": image_path,
                        "phash_hamming_distance": distance,
                        "ssim": round(similarity, 6),
                    })
                    break

        index = len(indexed_images)
        indexed_images.append((image_path, image_hash))
        for chunk in range(5):
            key = (chunk, (image_hash >> (chunk * 13)) & 0x1FFF)
            buckets[key].append(index)

    return duplicate_pairs
