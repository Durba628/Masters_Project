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
import cv2
model = YOLO("yolov8n.pt")

model.train(data="D:\worm_yolo_dataset\worm.yaml", epochs=40, freeze=10, project="D:\yolo_runs", name="worm_train")

results =model.predict(source="C:/Users/durba/Downloads/Durba_PM/Durba_PM/Agg/10 (10)_Export_RGB_FITC.tif", show=True, save=True, project="D:\yolo_runs", name="worm_train")

for r in results:
    img = r.plot(line_width=1, font_size=0.4, labels=False, conf=True)  
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    r.save()

#len(r.boxes)#total count
