import os
import numpy as np
from PIL import Image
import skimage.io as sio
from tqdm import tqdm
import torch
import torchvision
from torchvision.transforms import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou
from torch.amp import autocast, GradScaler

# ---------------------------
# Config
# ---------------------------
train_dir = "./train"
test_dir = "./test_release"
test_map_path = "./test_image_name_to_ids.json"
output_json_path = "./test-results.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 5  # background + class1~class4
MAX_INSTANCES = 50

# tensorboard logdir with class name
logdir_root = "runs/per_class"
os.makedirs(logdir_root, exist_ok=True)


# ---------------------------
# Dataset
# ---------------------------
class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, root, target_class, max_instances=MAX_INSTANCES):
        self.root = root
        self.imgs = list(sorted(os.listdir(root)))
        self.target_class = target_class
        self.max_instances = max_instances

    def __getitem__(self, idx):
        img_folder = os.path.join(self.root, self.imgs[idx])
        img_path = os.path.join(img_folder, "image.tif")
        img = Image.open(img_path).convert("RGB")

        masks = []
        labels = []

        mask_path = os.path.join(img_folder, f"class{self.target_class}.tif")
        if os.path.exists(mask_path):
            mask = sio.imread(mask_path)
            for inst_id in np.unique(mask):
                if inst_id == 0:
                    continue
                binary_mask = (mask == inst_id).astype(np.uint8)
                masks.append(binary_mask)
                labels.append(1)

        if len(masks) == 0:
            # ⭐如果沒有mask，直接產生一個假mask（極小不重要的假box，讓模型還能訓練）
            masks = torch.zeros((1, img.height, img.width), dtype=torch.uint8)
            boxes = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
            labels = torch.tensor([0], dtype=torch.int64)  # ⭐標成 background
        else:
            if len(masks) > self.max_instances:
                selected = np.random.choice(
                    len(masks), self.max_instances, replace=False
                )
                masks = [masks[i] for i in selected]
                labels = [labels[i] for i in selected]
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = []
            for mask in masks:
                pos = mask.nonzero()
                xmin, ymin = pos[:, 1].min(), pos[:, 0].min()
                xmax, ymax = pos[:, 1].max(), pos[:, 0].max()
                boxes.append(
                    [xmin.item(), ymin.item(), xmax.item(), ymax.item()]
                )
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }

        img = F.to_tensor(img)

        if np.random.rand() > 0.5:
            img = F.hflip(img)
            masks = masks.flip(-1)
            target["masks"] = masks
            target["boxes"][:, [0, 2]] = (
                img.shape[2] - target["boxes"][:, [2, 0]]
            )

        if np.random.rand() > 0.5:
            img = F.adjust_brightness(
                img, brightness_factor=np.random.uniform(0.8, 1.2)
            )

        if np.random.rand() > 0.5:
            img = F.adjust_contrast(
                img, contrast_factor=np.random.uniform(0.8, 1.2)
            )

        if np.random.rand() > 0.5:
            img = F.adjust_gamma(img, gamma=np.random.uniform(0.9, 1.1))
        return img, target

    def __len__(self):
        return len(self.imgs)


# ---------------------------
# AP50 計算
# ---------------------------
def compute_ap(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0
    ious = box_iou(pred_boxes, gt_boxes)
    matches = []
    for pred_idx in range(len(pred_boxes)):
        iou, gt_idx = ious[pred_idx].max(0)
        if iou >= iou_threshold and gt_idx.item() not in matches:
            matches.append(gt_idx.item())
    precision = len(matches) / (len(pred_boxes) + 1e-6)
    recall = len(matches) / (len(gt_boxes) + 1e-6)
    if precision + recall == 0:
        return 0.0
    return (precision * recall) / (precision + recall)


# ---------------------------
# Train Function
# ---------------------------
def train_per_class(target_class):
    dataset = MedicalDataset(train_dir, target_class=target_class)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x)),
    )

    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT"
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, 2
        )
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = (
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, hidden_layer, 2
        )
    )

    for name, parameter in model.backbone.body.named_parameters():
        if "layer4" not in name:
            parameter.requires_grad = False

    model.to(device)

    params = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": 1e-5,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": 1e-4,
        },
    ]

    num_epochs = 300
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    scaler = GradScaler()

    writer = SummaryWriter(
        log_dir=os.path.join(logdir_root, f"class{target_class}")
    )

    for epoch in range(num_epochs):
        model.train()
        pbar = tqdm(
            loader, desc=f"[class{target_class}] Epoch {epoch+1}/{num_epochs}"
        )
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [
                {k: v.to(device) for k, v in t.items()} for t in targets
            ]
            optimizer.zero_grad()
            with autocast("cuda"):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Eval AP
        model.eval()
        ap_sum = 0.0
        with torch.no_grad():
            for images, targets in loader:
                images = [img.to(device) for img in images]
                targets = [
                    {k: v.to(device) for k, v in t.items()} for t in targets
                ]
                predictions = model(images)
                for pred, gt in zip(predictions, targets):
                    ap = compute_ap(pred["boxes"], pred["scores"], gt["boxes"])
                    ap_sum += ap
        ap_mean = ap_sum / len(loader)
        writer.add_scalar("AP50/train", ap_mean, epoch)
        print(f"AP50 (class{target_class}, epoch {epoch+1}): {ap_mean:.4f}")

    writer.close()
    torch.save(model.state_dict(), f"class{target_class}_model.pth")


# ---------------------------
# Main: train all 4 classes
# ---------------------------
if __name__ == "__main__":
    for cls in range(1, 5):
        train_per_class(cls)
