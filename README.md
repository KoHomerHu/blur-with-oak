# Blur with OAK

Real-time face detection + tracking + blurring using OAK-D-Lite. We use the [`yolov8n-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt) model from the [yolov8-face](https://github.com/akanametov/yolov8-face/tree/dev) repo and the implementation of SORT from the [haroonshakeel/yolov7-object-tracking](https://github.com/haroonshakeel/yolov7-object-tracking) repo.

This is similar to the DepthAI experiment [gen2-blur-faces](https://github.com/luxonis/depthai-experiments/tree/master/gen2-blur-faces), although we use YOLOv8n and the SORT algorithm instead of mobilenet detection and object tracking on OAK. 
