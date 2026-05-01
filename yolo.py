#from ultralytics import YOLO

#model = YOLO("yolov8n.pt")  
#model = YOLO("C:/Users/durba/python/yolov8n.pt")

#for param in model.model.parameters():
    #param.requires_grad = False
 
#model.train(data="coco8.yaml", epochs=10) 


#results = model("C:/Users/durba/Downloads/test")  # test folder of images
#results.show()  # display bounding boxes
#results.save("runs/detect")  # save results with boxes


#transfer learning with partial freezing
from ultralytics import YOLO
import torch


print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU:  {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

dataset_yaml = r"C:/Count My Proteins/yolo_train_dataset/worm.yaml"


print("\nStage 1: Frozen backbone...")
model = YOLO('yolov8m.pt')

model.train(
    data          = dataset_yaml,
    epochs        = 50,
    imgsz         = 1024,
    batch         = 8,
    #patience      = 20,

    optimizer     = 'AdamW',
    lr0           = 0.001,
    lrf           = 0.01,
    weight_decay  = 0.0005,
    warmup_epochs = 3,

    freeze        = 10,

    # All augmentation off
    mosaic        = 0.0,
    mixup         = 0.0,
    copy_paste    = 0.0,
    hsv_h         = 0.0,
    hsv_s         = 0.0,
    hsv_v         = 0.0,
    degrees       = 0.0,
    translate     = 0.0,
    scale         = 0.0,
    fliplr        = 0.0,
    flipud        = 0.0,

    device        = 0,
    workers       = 8,
    amp           = True,
    project       = r'C:/Count My Proteins/runs',
    name          = 'stage1_frozen',
    exist_ok      = True,
    save          = True,
    save_period   = 10,
    verbose       = True,
)


print("\nStage 2: Full fine-tune, early stopping patience=10...")
model = YOLO(r'C:/Count My Proteins/runs/stage1_frozen/weights/best.pt')

model.train(
    data          = dataset_yaml,
    epochs        = 1000,
    imgsz         = 1024,
    batch         = 8,
    #patience      = 10,

    optimizer     = 'AdamW',
    lr0           = 0.0001,
    lrf           = 0.001,
    weight_decay  = 0.0005,
    warmup_epochs = 2,

    freeze        = 0,

    # All augmentation off
    mosaic        = 0.0,
    mixup         = 0.0,
    copy_paste    = 0.0,
    hsv_h         = 0.0,
    hsv_s         = 0.0,
    hsv_v         = 0.0,
    degrees       = 0.0,
    translate     = 0.0,
    scale         = 0.0,
    fliplr        = 0.0,
    flipud        = 0.0,

    device        = 0,
    workers       = 8,
    amp           = True,
    project       = r'C:/Count My Proteins/runs',
    name          = 'stage2_finetune',
    exist_ok      = True,
    save          = True,
    save_period   = 10,
    verbose       = True,
)


# print("\nEvaluating on test set...")
# model = YOLO(r'C:\Count My Proteins\runs\stage2_finetune\weights\best.pt')

# metrics = model.val(
#     data    = dataset_yaml,
#     split   = 'test',
#     imgsz   = 1024,
#     batch   = 1,
#     device  = 0,
#     verbose = True,
# )

# print(f"\nmAP50:     {metrics.box.map50:.4f}")
# print(f"mAP50-95:  {metrics.box.map:.4f}")
# print(f"Precision: {metrics.box.mp:.4f}")
# print(f"Recall:    {metrics.box.mr:.4f}")        
