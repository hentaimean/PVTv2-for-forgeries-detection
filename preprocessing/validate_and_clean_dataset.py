import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Подавляем шумные EXIF-предупреждения
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")


def validate_pair(
    img_path: Path,
    mask_path: Path,
    check_mask_binary: bool = True,
    check_size_match: bool = True,
    check_mask_grayscale: bool = True,
    mask_expected_values: set = {0, 255}
):
    """
    Validates and virtually normalizes image-mask pair as it would be loaded for training.
    Returns True if valid, False if should be removed.
    """
    try:
        # === Загружаем и нормализуем изображение ===
        with Image.open(img_path) as im:
            # Приводим к RGB — стандарт для моделей
            img = im.convert('RGB')

        # === Загружаем и нормализуем маску ===
        with Image.open(mask_path) as im:
            # Приводим к grayscale (L)
            mask = im.convert('L')

        # === Проверки ===
        if check_size_match:
            if img.size != mask.size:
                return False

        if check_mask_grayscale:
            # После convert('L') это всегда True, но проверяем исходный режим на всякий случай
            original_mask = Image.open(mask_path)
            if original_mask.mode not in ('L', '1'):
                # Но даже если был RGB — мы привели к L, так что это не причина для удаления
                # Поэтому эту проверку можно считать "информационной", а не фатальной
                pass  # не удаляем!

        if check_mask_binary:
            mask_arr = np.array(mask, dtype=np.uint8)
            unique_vals = set(np.unique(mask_arr))
            if not unique_vals.issubset(mask_expected_values):
                return False

        return True

    except Exception:
        # Любая ошибка открытия или обработки → удаляем
        return False


def process_pair(args):
    img_path, mask_path, checks = args
    is_valid = validate_pair(
        img_path=img_path,
        mask_path=mask_path,
        check_mask_binary=checks.get("mask_binary", True),
        check_size_match=checks.get("size_match", True),
        check_mask_grayscale=checks.get("mask_grayscale", True),
        mask_expected_values=checks.get("mask_values", {0, 255})
    )
    if not is_valid:
        try:
            if img_path.exists():
                os.remove(img_path)
            if mask_path.exists():
                os.remove(mask_path)
        except Exception:
            pass  # игнорируем ошибки удаления
        return False
    return True


def main(images_dir: str, masks_dir: str, num_workers: int = 12, checks: dict = None):
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError("One or both directories do not exist.")

    # Сопоставление по имени без расширения
    image_files = {f.stem: f for f in images_dir.iterdir() if f.is_file()}
    mask_files = {f.stem: f for f in masks_dir.iterdir() if f.is_file()}

    common_stems = set(image_files.keys()) & set(mask_files.keys())

    if not common_stems:
        print("⚠️ No matching pairs found (by stem).")
        return

    print(f"📁 Images: {len(image_files)}, Masks: {len(mask_files)}")
    print(f"🔗 Matching pairs: {len(common_stems)}")

    task_args = [
        (image_files[stem], mask_files[stem], checks or {})
        for stem in common_stems
    ]

    valid_count = 0
    total = len(task_args)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_pair, args) for args in task_args]
        for future in tqdm(as_completed(futures), total=total, desc="Validating"):
            if future.result():
                valid_count += 1

    print(f"\n✅ Kept {valid_count} valid pairs.")
    print(f"❌ Removed {total - valid_count} invalid pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate dataset for segmentation training.")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--masks", required=True, help="Path to masks directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--no-mask-binary-check", action="store_true", help="Skip binary mask check")
    parser.add_argument("--no-size-check", action="store_true", help="Skip size matching check")
    parser.add_argument("--no-mask-grayscale-check", action="store_true", help="Skip grayscale mask check (not recommended)")
    parser.add_argument("--mask-values", type=str, default="0,255",
                        help="Allowed pixel values in masks (e.g., '0,1' or '0,255')")

    args = parser.parse_args()

    mask_vals = set(int(v.strip()) for v in args.mask_values.split(","))

    checks = {
        "mask_binary": not args.no_mask_binary_check,
        "size_match": not args.no_size_check,
        "mask_grayscale": not args.no_mask_grayscale_check,
        "mask_values": mask_vals
    }

    main(
        images_dir=args.images,
        masks_dir=args.masks,
        num_workers=args.workers,
        checks=checks
    )