My masters project is building ml model for automated counting of protein aggregates in fluoroscent c.elegans images. 
The trial_model_one.m is the built from scratch convolutional neural network model on matlab, it works on small dataset where feature extraction and the job of classification is rather simple.
The try_count_two.m is an automated method of aggregate counting on matlab, that applies boundary boxes on the aggregates for tranparency of the results.
The yolo.py is the yolov8n code on python that does detection box prediction using tranfer learning and partially freezing the first 10 layers. Libraries used were (Yolov8) Ultralytics and cv2.
