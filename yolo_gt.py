## COMBINED IMAGES
from ultralytics import YOLO
import cv2
import os

model = YOLO("C:/Users/hp/runs/detect/train3/weights/best.pt")
img_folder = "D:/Count My Proteins/yolo_train_dataset/images/val"
label_folder = "D:/Count My Proteins/yolo_train_dataset/labels/val"
output_folder = "D:/combined_images"

os.makedirs(output_folder, exist_ok=True)

img_files = sorted(os.listdir(img_folder))

saved_count = 0
max_images_to_save = 10

for img_name in img_files:
    if saved_count >= max_images_to_save:
        break

    img_path = os.path.join(img_folder, img_name)

    results = model.predict(img_path, conf=0.02, save=False)
    r = results[0]
    pred_img = r.plot(line_width=2)

    if pred_img is None:
        continue

    pred_count = 0
    areas = []
    
    if r.boxes is not None and len(r.boxes) > 0:
        pred_count = len(r.boxes)
        
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = box.tolist()
            width = x2 - x1
            height = y2 - y1
            area = width * height
            areas.append(area)
    
    if areas:
        mean_size = sum(areas) / len(areas)
        min_size = min(areas)
        max_size = max(areas)
        mean_side = sum((((x2-x1)+(y2-y1))/2) for box in r.boxes.xyxy 
                        for x1,y1,x2,y2 in [box.tolist()]) / len(areas)
        um_per_px = 0.75 / mean_side          # Q35 ~0.75 µm
        nm_per_px = um_per_px * 1000
    else:
        mean_size = 0
        min_size = 0
        max_size = 0
        nm_per_px = 0

    gt_img = cv2.imread(img_path)
    if gt_img is None:
        continue

    pred_h, pred_w = pred_img.shape[:2]
    gt_h, gt_w = gt_img.shape[:2]
    if pred_h != gt_h:
        scale = gt_h / pred_h
        pred_img = cv2.resize(pred_img, (int(pred_w * scale), gt_h))

    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_folder, label_name)
    gt_count = sum(1 for _ in open(label_path))

    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())
                x1 = int((x - bw / 2) * gt_w)
                y1 = int((y - bh / 2) * gt_h)
                x2 = int((x + bw / 2) * gt_w)
                y2 = int((y + bh / 2) * gt_h)
                x1 = max(0, min(gt_w, x1))
                y1 = max(0, min(gt_h, y1))
                x2 = max(0, min(gt_w, x2))
                y2 = max(0, min(gt_h, y2))
                cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    y_offset = 300
    line_height = 60
    font_scale = 1
    thickness = 2
    color = (255, 255, 255)
    color2=(200,200,200)
    
    cv2.putText(pred_img, f"Count: {pred_count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    cv2.putText(pred_img, f"Mean Size: {mean_size:.0f} px", (10, y_offset + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    cv2.putText(pred_img, f"Min Size: {min_size:.0f} px", (10, y_offset + 2 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    cv2.putText(pred_img, f"Max Size: {max_size:.0f} px", (10, y_offset + 3 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

   
    cv2.putText(pred_img, f"1px = {nm_per_px:.2f} nm (Q35~0.75um)", (10, y_offset + 10 * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color2, thickness)

    cv2.putText(gt_img, f"GT: {gt_count}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    combined = cv2.hconcat([pred_img, gt_img])

    output_path = os.path.join(output_folder, os.path.splitext(img_name)[0] + "_combined_new_Final.png")
    cv2.imwrite(output_path, combined)
    
    saved_count += 1

print(f"Saved {saved_count} combined images to: {output_folder}")


## CONFUSION MATRIX

if __name__ == '__main__':
    from ultralytics import YOLO
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import os

    
    MODEL_PATH = "C:/Users/hp/runs/detect/train3/weights/best.pt"
    DATA_YAML  = "D:/Count My Proteins/yolo_train_dataset/worm.yaml"  # ← point to your data.yaml
    SAVE_DIR="C:/Users/hp/runs/detect/train3/Evaluation"
    CONF        = 0.5
    IOU         = 0.5

    
    model = YOLO(MODEL_PATH)

    results = model.val(
        data=DATA_YAML,
        conf=0.5,
        iou=0.5,
        imgsz=1280,       
        augment=True,      
        max_det=1000,      
        agnostic_nms=True,
        verbose=True
    )

    
    precision = results.box.mp      # mean precision
    recall    = results.box.mr      # mean recall
    map50     = results.box.map50   # mAP@0.5
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

   
    cm = results.confusion_matrix 
    matrix = cm.matrix             

   
    print("\nRaw YOLO Confusion Matrix:")
    print(matrix)

   
    TP = int(matrix[0, 0])
    FN = int(matrix[1, 0])   # predicted background, actually object
    FP = int(matrix[0, 1])   # predicted object, actually background

    print(f"\nTP: {TP} | FP: {FP} | FN: {FN}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"mAP@0.5   : {map50:.4f}")
    print(f"F1        : {f1:.4f}")

   
    calc_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    calc_recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    print(f"\nVerification from counts:")
    print(f"Precision: {calc_precision:.4f}  (reported: {precision:.4f})")
    print(f"Recall   : {calc_recall:.4f}  (reported: {recall:.4f})")

    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    
    ax = axes[0]
    valid = np.array([[TP, FN], [FP, 0]], dtype=float)
    im = ax.imshow(valid, cmap="Blues", vmin=0)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nObject", "Predicted\nNothing"], fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actual Object", "Actual Nothing"], fontsize=10)

    cell_labels = {
        (0, 0): f"TP\n{TP:,}",
        (0, 1): f"FN\n{FN:,}",
        (1, 0): f"FP\n{FP:,}",
        (1, 1): "TN\n(N/A)",
    }
    for (i, j), text in cell_labels.items():
        if i == 1 and j == 1:
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, color="#dddddd", zorder=2))
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=10, color="#888888", zorder=3)
        else:
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color="white", zorder=3)

    plt.colorbar(im, ax=ax, label="Count")
    ax.set_title("Detection Confusion Matrix\n(from YOLO evaluator)", fontsize=11)

   
    ax2 = axes[1]
    metrics      = [precision, recall, map50, f1]
    metric_names = ["Precision", "Recall", "mAP@0.5", "F1"]
    colors       = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    bars = ax2.bar(metric_names, metrics, color=colors, width=0.5, edgecolor="white")
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Score", fontsize=11)
    ax2.set_title("Model Performance Metrics", fontsize=11)
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 baseline")

    for bar, val in zip(bars, metrics):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Single-Class Object Detection Evaluation  |  conf={0.5}  iou={0.5}",
                fontsize=12, fontweight="bold", y=1.01)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"detection_evaluation_at_conf {CONF}_at_iou {IOU}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close() 

   
    if os.path.exists(save_path):
        size_kb = os.path.getsize(save_path) / 1024
        print(f"\nPlot saved successfully!")
        print(f"   Path : {save_path}")
        print(f"   Size : {size_kb:.1f} KB")
    else:
        print(f"\n Save FAILED — file not found at: {save_path}")
