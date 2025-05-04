import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from pycocotools import mask as mask_util
from torchvision.ops import nms
import zipfile
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ---------------------------
# Config
# ---------------------------
test_dir = "./test_release"
test_map_path = "./test_image_name_to_ids.json"
output_json_path = "./test-results.json"
zip_output_path = "submission.zip"
score_threshold = 0.01

model_paths = {
    1: "class1_model.pth",
    2: "class2_model.pth",
    3: "class3_model.pth",
    4: "class4_model.pth",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# RLE Encode
# ---------------------------
def encode_mask(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


# ---------------------------
# Load All Models Once
# ---------------------------
def load_all_models():
    models = {}
    for cls_id, path in model_paths.items():
        model = maskrcnn_resnet50_fpn_v2(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
        mask_feat = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_feat, 256, 2)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        models[cls_id] = model
    return models


# ---------------------------
# Inference with Merged NMS
# ---------------------------
def inference_all_models():
    with open(test_map_path, "r") as f:
        name_to_id_list = json.load(f)
    name_to_id = {e["file_name"]: e["id"] for e in name_to_id_list}
    test_imgs = sorted(os.listdir(test_dir))
    models = load_all_models()

    all_outputs = []

    for img_file in tqdm(test_imgs, desc="Inference"):
        img_path = os.path.join(test_dir, img_file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        image_id = name_to_id[img_file]

        combined_boxes, combined_scores, combined_labels, combined_masks = (
            [],
            [],
            [],
            [],
        )

        for cls_id, model in models.items():
            with torch.no_grad():
                pred = model([img_tensor])[0]

            for box, score, label, mask in zip(
                pred["boxes"], pred["scores"], pred["labels"], pred["masks"]
            ):
                if score < score_threshold or label.item() != 1:
                    continue
                combined_boxes.append(box)
                combined_scores.append(score)
                combined_labels.append(cls_id)
                combined_masks.append(mask)

        if len(combined_boxes) == 0:
            continue

        boxes = torch.stack(combined_boxes)
        scores = torch.tensor(combined_scores).to(device)
        masks = combined_masks
        labels = combined_labels

        keep = nms(boxes, scores, iou_threshold=0.5)
        for idx in keep:
            box = boxes[idx].cpu().numpy()
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min
            mask_bin = masks[idx].squeeze().cpu().numpy() > 0.5
            segmentation = {
                "size": list(mask_bin.shape),
                "counts": encode_mask(mask_bin)["counts"],
            }
            all_outputs.append(
                {
                    "image_id": int(image_id),
                    "bbox": [
                        float(x_min),
                        float(y_min),
                        float(width),
                        float(height),
                    ],
                    "score": float(scores[idx].item()),
                    "category_id": int(labels[idx]),
                    "segmentation": segmentation,
                }
            )

    with open(output_json_path, "w") as f:
        json.dump(all_outputs, f, indent=4)
    print(f"Saved {len(all_outputs)} predictions to {output_json_path}")


# ---------------------------
# Zip Result
# ---------------------------
def make_submission_zip():
    with zipfile.ZipFile(zip_output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(output_json_path, arcname="test-results.json")
    print(f"Submission zip created: {zip_output_path}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    inference_all_models()
    make_submission_zip()
