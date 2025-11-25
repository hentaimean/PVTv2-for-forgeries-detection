# preprocessing/validate_and_clean_dataset.py — валидация и очистка датасета для сегментации

import argparse
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================
# КОНСТАНТЫ ПРОЕКТА
# ==============================

# Параметры по умолчанию
DEFAULT_NUM_WORKERS = 8
DEFAULT_MASK_VALUES = {0, 1}

# Подавление шумных EXIF-предупреждений
warnings.filterwarnings("ignore", message="Corrupt EXIF data.*")


# ==============================
# ФУНКЦИИ ВАЛИДАЦИИ
# ==============================

def validate_pair(
        img_path: Path,
        mask_path: Path,
        check_mask_binary: bool = True,
        check_size_match: bool = True,
        check_mask_grayscale: bool = True,
        mask_expected_values: set = DEFAULT_MASK_VALUES
):
    """
    Проверяет корректность пары изображение-маска, как она будет загружаться при обучении.

    Аргументы:
        img_path (Path): путь к изображению.
        mask_path (Path): путь к маске.
        check_mask_binary (bool): проверять, что маска бинарная.
        check_size_match (bool): проверять совпадение размеров.
        check_mask_grayscale (bool): проверять, что маска в grayscale (информативно, но не критично).
        mask_expected_values (set): допустимые значения пикселей в маске.

    Возвращает:
        bool: True, если пара валидна; False — если её следует удалить.
    """
    try:
        # Загрузка и приведение изображения к RGB
        with Image.open(img_path) as im:
            img = im.convert('RGB')

        # Загрузка и приведение маски к grayscale
        with Image.open(mask_path) as im:
            mask = im.convert('L')

        # Проверка совпадения размеров
        if check_size_match and img.size != mask.size:
            return False

        # Проверка бинарности маски
        if check_mask_binary:
            mask_arr = np.array(mask, dtype=np.uint8)
            unique_vals = set(np.unique(mask_arr))
            if not unique_vals.issubset(mask_expected_values):
                return False

        # Примечание: проверка grayscale не влияет на решение об удалении,
        # так как mask = im.convert('L') всегда даёт корректный формат.
        return True

    except Exception:
        # Любая ошибка при открытии или обработке → пара недействительна
        return False


def process_pair(args):
    """
    Обрабатывает одну пару изображение-маска: проверяет и удаляет, если невалидна.

    Аргументы:
        args (tuple): (img_path, mask_path, checks_dict)

    Возвращает:
        bool: True, если пара осталась; False — если удалена.
    """
    img_path, mask_path, checks = args
    is_valid = validate_pair(
        img_path=img_path,
        mask_path=mask_path,
        check_mask_binary=checks.get("mask_binary", True),
        check_size_match=checks.get("size_match", True),
        check_mask_grayscale=checks.get("mask_grayscale", True),
        mask_expected_values=checks.get("mask_values", DEFAULT_MASK_VALUES)
    )

    # Удаление невалидных файлов
    if not is_valid:
        try:
            if img_path.exists():
                os.remove(img_path)
            if mask_path.exists():
                os.remove(mask_path)
        except Exception:
            # Игнорируем ошибки удаления (например, файл уже удалён)
            pass
        return False

    return True


# ==============================
# ОСНОВНАЯ ФУНКЦИЯ
# ==============================

def main(images_dir: str, masks_dir: str, num_workers: int = DEFAULT_NUM_WORKERS, checks: dict = None):
    """
    Основная функция валидации и очистки датасета.

    Аргументы:
        images_dir (str): путь к папке с изображениями.
        masks_dir (str): путь к папке с масками.
        num_workers (int): число потоков для параллельной обработки.
        checks (dict): словарь настроек проверок.
    """
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.exists() or not masks_dir.exists():
        raise ValueError("Одна или обе директории не существуют.")

    # Сопоставление файлов по имени (без расширения)
    image_files = {f.stem: f for f in images_dir.iterdir() if f.is_file()}
    mask_files = {f.stem: f for f in masks_dir.iterdir() if f.is_file()}
    common_stems = set(image_files.keys()) & set(mask_files.keys())

    if not common_stems:
        print("Не найдено ни одной пары изображение-маска (по имени файла).")
        return

    print(f"Изображений: {len(image_files)}, Масок: {len(mask_files)}")
    print(f"Найдено пар: {len(common_stems)}")

    # Формирование задач для потоков
    task_args = [
        (image_files[stem], mask_files[stem], checks or {})
        for stem in common_stems
    ]

    # Параллельная обработка
    valid_count = 0
    total = len(task_args)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_pair, args) for args in task_args]
        for future in tqdm(as_completed(futures), total=total, desc="Валидация датасета"):
            if future.result():
                valid_count += 1

    print(f"\nСохранено {valid_count} валидных пар.")
    print(f"Удалено {total - valid_count} невалидных пар.")


# ==============================
# ТОЧКА ВХОДА
# ==============================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Валидация и очистка датасета для обучения сегментации.")

    parser.add_argument("--images", required=True, help="Путь к папке с изображениями")
    parser.add_argument("--masks", required=True, help="Путь к папке с масками")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS, help="Число потоков")
    parser.add_argument("--no-mask-binary-check", action="store_true", help="Отключить проверку бинарности маски")
    parser.add_argument("--no-size-check", action="store_true", help="Отключить проверку совпадения размеров")
    parser.add_argument("--no-mask-grayscale-check", action="store_true",
                        help="Отключить проверку формата маски (не рекомендуется)")
    parser.add_argument(
        "--mask-values",
        type=str,
        default=",".join(map(str, DEFAULT_MASK_VALUES)),
        help="Допустимые значения пикселей в масках (например: '0,1' или '0,255')"
    )

    args = parser.parse_args()

    # Парсинг допустимых значений маски
    mask_vals = set(int(v.strip()) for v in args.mask_values.split(","))

    # Настройка проверок
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