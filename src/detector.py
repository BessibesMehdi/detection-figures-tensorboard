"""Détecteur minimal : feature maps -> prédiction par cellule -> NMS -> une boîte.

Image 64x64 -> 3 blocs conv stride 2 -> feature map 32x8x8 (stride 8).
Une conv 1x1 prédit pour chaque cellule 5 valeurs : (objectness, dx, dy, w, h)
  - dx, dy in [0,1] : décalage du centre dans la cellule
  - w, h in [0,1]   : taille relative à l'image
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

STRIDE = 8

def iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU entre une boîte a (4,) et des boîtes b (N, 4)."""
    inter_w = (torch.minimum(a[2], b[:, 2]) - torch.maximum(a[0], b[:, 0])).clamp(min=0)
    inter_h = (torch.minimum(a[3], b[:, 3]) - torch.maximum(a[1], b[:, 1])).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 1.1) -> list[int]:
    """Garde la meilleure boîte, supprime celles qui la chevauchent, recommence et retourne les indices des boîtes conservées."""
    order = scores.argsort(descending=True)
    kept = []
    while len(order):
        best = order[0]
        kept.append(int(best))
        order = order[1:]
        order = order[iou(boxes[best], boxes[order]) < iou_threshold]
    return kept


def code(cx, cy, w, h, class_id=0, size=64) -> torch.Tensor:
    """Boîte [cx, cy, w, h, class_id] en pixels -> (8, G, G) pour l'entraînement."""
    grid = size // STRIDE
    gx, gy = cx / size * grid, cy / size * grid
    col, row = min(int(gx), grid - 1), min(int(gy), grid - 1)
    target = torch.zeros(8, grid, grid)

    target[0:5, row, col] = torch.tensor([1.0, gx - col, gy - row, w / size, h / size])
    
    # Allume un 1 dans la case de la bonne forme et on ajoute les nouvelles formes à apprendre:
    # case 5 si cercle (classe 0), case 6 si carré (classe 1), case 7 si triangle (classe 2)
    if class_id in (0, 5, "0", "5", "circle", "cercle"):
        target[5, row, col] = 1.0
    elif class_id in (1, 6, "1", "6", "square", "carré", "carre"):
        target[6, row, col] = 1.0
    elif class_id in (2, 7, "2", "7", "triangle"):
        target[7, row, col] = 1.0
    elif 0 <= int(class_id) <= 2:
        target[5 + int(class_id), row, col] = 1.0
    elif 5 <= int(class_id) <= 7:
        target[int(class_id), row, col] = 1.0
    return target

def decode(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(8, G, G) -> boîtes (G*G, 4) en pixels [x1, y1, x2, y2], scores (G*G,) et classes (G*G,)."""
    grid = logits.shape[-1]
    p = torch.sigmoid(logits)
    rows, cols = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    cx = (cols + p[1]) * STRIDE
    cy = (rows + p[2]) * STRIDE
    w, h = p[3] * size, p[4] * size
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    classes = logits[5:8].argmax(dim=0)
    return boxes.reshape(-1, 4), p[0].reshape(-1), classes.reshape(-1)



class ShapeDataset(Dataset):
    """Cible (5, G, G) : tout à 0 sauf la cellule contenant le centre."""

    def __init__(self, folder: Path, size: int = 64) -> None:
        self.paths = sorted(Path(folder).glob("*.png"))
        if not self.paths:
            raise FileNotFoundError(f"Aucune image dans {folder} (lancer dataset_generator).")
        self.grid = size // STRIDE

    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[i]
        image = torch.from_numpy(
            np.asarray(Image.open(path).convert("RGB"), dtype=np.float32).copy()
        ).permute(2, 0, 1) / 255.0
        labels = list(map(float, path.with_suffix(".txt").read_text().split()))
        cx, cy, w, h = labels[:4]
        class_id = labels[4] if len(labels) > 4 else 0
        target = code(cx, cy, w, h, class_id, size=image.shape[-1])
        return image, target


class Detector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        blocks = [nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
                  nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                  nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU()]
        self.backbone = nn.Sequential(*blocks)
        self.head = nn.Conv2d(32, 8, kernel_size=1, stride = 1)#IL FAUT COMPTER LES 5 FEATURE (p,dx,dy,w,h) et aussi 3 autres pour savoir si c'est carré cercle ou triangle donc on passe de 5 à 8 
        # A_COMPLETER Aide : Définir les couches du détecteur (backbone et head).
        

    def feature_maps(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.backbone(images)
        # A_COMPLETER Aide : Appliquer le backbone sur les images pour obtenir les feature maps.
        
        return feature_maps

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.feature_maps(images)
        output = self.head(feature_maps)
        # A_COMPLETER Aide : Appliquer la tête sur les feature maps pour obtenir les logits de sortie.
        
        return output





def compute_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE sur l'objectness de toutes les cellules + MSE sur la boîte + Cross-Entropy sur la forme de la cellule positive."""
    obj_loss = nn.functional.binary_cross_entropy_with_logits(
        logits[:, 0], target[:, 0], pos_weight=torch.tensor(10.0))
    positive = target[:, 0] > 0.5  # (B, G, G)
    pred_box = torch.sigmoid(logits[:, 1:5]).permute(0, 2, 3, 1)[positive]
    true_box = target[:, 1:5].permute(0, 2, 3, 1)[positive]
    box_loss = 10.0 * nn.functional.mse_loss(pred_box, true_box)

    pred_cls = logits[:, 5:].permute(0, 2, 3, 1)[positive]
    true_cls = target[:, 5:].permute(0, 2, 3, 1)[positive]
    cls_loss = nn.functional.cross_entropy(pred_cls, true_cls)

    return obj_loss + box_loss + cls_loss




def predict(logits: torch.Tensor, size: int = 64) -> tuple[torch.Tensor, float, int]:
    """Une seule figure par image : la boîte de plus fort score après NMS, son score et sa forme."""
    boxes, scores, classes = decode(logits, size)
    best = nms(boxes, scores)[0]
    return boxes[best].detach(), float(scores[best].detach()), int(classes[best].detach())
