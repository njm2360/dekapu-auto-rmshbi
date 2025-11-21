import cv2
import logging
import numpy as np
from pathlib import Path

Size = tuple[int, int]


class MaskLoader:

    @classmethod
    def load(cls, mask_path: Path, size: Size) -> np.ndarray:
        if not mask_path.exists():
            logging.info("Mask file not found. Using default mask")
            return cls.make_default()

        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            logging.warning(f"Failed to load mask image. Using default mask")
            return cls.make_default()

        h, w = mask_gray.shape[:2]

        if (w, h) != size:
            logging.warning(
                f"Mask size {w}x{h} does not match WINDOW_SIZE {size}. "
                "Using default mask"
            )
            return cls.make_default()

        _, mask_bin = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        logging.info("Mask image load successfully")
        return mask_bin

    @classmethod
    def make_default(cls, size: Size) -> np.ndarray:
        w, h = size
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2 :, :] = 255
        return mask
