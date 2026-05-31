import torch
import numpy as np
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2
from torch.utils.data import Dataset, random_split
from PIL import Image


# ── 상수 ─────────────────────────────────────────────────────
RESIZE        = (384, 384)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Transform 팩토리 ─────────────────────────────────────────
def make_train_transforms():
    """Train: Resize → RandomAffine → HFlip → ColorJitter → GaussianBlur → Normalize.
    RandomAffine(scale=0.9~1.1)으로 경계가 잘리지 않는 안전한 스케일 변화를 주고,
    RandomResizedCrop 대신 사용해 boundary 정보를 보존합니다.
    v2.Compose 에 (image, mask) 쌍을 넣으면 공간 변환은 동시에 적용되고,
    ColorJitter/GaussianBlur 는 tv_tensors.Mask 를 인식해 이미지에만 적용됩니다.
    """
    return v2.Compose([
        v2.Resize(RESIZE),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomAffine(degrees=15, scale=(0.9, 1.1), translate=(0.05, 0.05)),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_val_transforms():
    """Val / Test: Resize(384) + Normalize (augmentation 없음)."""
    return v2.Compose([
        v2.Resize(RESIZE),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── 원본 PIL 쌍을 반환하는 베이스 Dataset ────────────────────
class _RawPetSegDataset(Dataset):
    """
    (PIL Image, uint8 numpy mask) 쌍을 반환합니다.
    trimap 픽셀값 {1,2,3} → 제출 규약 {0,1,2} 로 remapping (값 - 1).
    """
    def __init__(self, root, split, download=True):
        self._base = datasets.OxfordIIITPet(
            root=root,
            split=split,
            download=download,
            target_types='segmentation',
            transform=None,
            target_transform=None,
        )

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        image_pil, mask_pil = self._base[idx]
        mask_np = (np.array(mask_pil, dtype=np.int16) - 1).clip(0, 2).astype(np.uint8)
        return image_pil, mask_np


class _CombinedRaw(Dataset):
    """두 _RawPetSegDataset 을 이어 붙입니다."""
    def __init__(self, d1, d2):
        self._d1, self._d2 = d1, d2

    def __len__(self):
        return len(self._d1) + len(self._d2)

    def __getitem__(self, idx):
        if idx < len(self._d1):
            return self._d1[idx]
        return self._d2[idx - len(self._d1)]


class _TransformDataset(Dataset):
    """
    베이스 Dataset 의 부분집합에 image+mask 동시 transform 을 적용합니다.
    image → tv_tensors.Image, mask → tv_tensors.Mask 로 감싸므로
    v2 transform 이 각 타입에 맞게 동작합니다.
    """
    def __init__(self, base, indices, transforms):
        self._base       = base
        self._indices    = list(indices)
        self._transforms = transforms

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, i):
        image_pil, mask_np = self._base[self._indices[i]]

        img_tensor  = tv_tensors.Image(v2.functional.to_image(image_pil))  # CHW uint8
        mask_tensor = tv_tensors.Mask(mask_np)                              # HW uint8

        img_tensor, mask_tensor = self._transforms(img_tensor, mask_tensor)

        # mask 는 class index 용 long tensor (H, W)
        return img_tensor, mask_tensor.squeeze(0).long()


# ── 데이터셋 구성 ─────────────────────────────────────────────
data_dir = './oxford_pet_data'

# 원본(다운로드) 데이터셋은 seed와 무관하므로 모듈 로드 시 한 번만 구성합니다.
_raw_trainval = _RawPetSegDataset(data_dir, split='trainval', download=True)
_raw_test     = _RawPetSegDataset(data_dir, split='test',     download=True)
_full_raw     = _CombinedRaw(_raw_trainval, _raw_test)


def build_loaders(seed: int = 42, batch_size: int = 16, num_workers: int = 4):
    """주어진 seed로 train/val(90:10) 분할을 만들고 DataLoader 를 반환합니다.

    분할은 seed 전용 Generator 로 재현 가능하게 구성하므로, seed 가 다르면
    서로 다른 train/val 구성을 얻어 앙상블 멤버 간 다양성이 생깁니다.

    Returns
    -------
    (train_loader, val_loader, train_size, val_size)
    """
    total_size = len(_full_raw)
    train_size = int(0.9 * total_size)
    val_size   = total_size - train_size

    split_gen = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(
        range(total_size), [train_size, val_size], generator=split_gen
    )

    train_set = _TransformDataset(_full_raw, train_idx, make_train_transforms())
    val_set   = _TransformDataset(_full_raw, val_idx,   make_val_transforms())

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, len(train_set), len(val_set)
