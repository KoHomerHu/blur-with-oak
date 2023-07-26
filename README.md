# Blur with OAK

Real-time face detection + tracking + blurring using OAK-D-Lite. We use the [`yolov8n-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt) model from the [yolov8-face](https://github.com/akanametov/yolov8-face/tree/dev) repo and the implementation of SORT is adapted from the [haroonshakeel/yolov7-object-tracking](https://github.com/haroonshakeel/yolov7-object-tracking) repo. The script can be run on edge devices (e.g. Raspberry Pi) without a significant drop in FPS (near 35 which is the capped maximum on OAK-D-Lite).

What we are doing is similar to the DepthAI experiment [gen2-blur-faces](https://github.com/luxonis/depthai-experiments/tree/master/gen2-blur-faces), although we use YOLOv8n and the SORT algorithm instead of mobilenet detection and zero term object tracking node on OAK. 

## Argument `keyboardOn`

Run the following command to activate the keyboard switch for activating/deactivating blurring: 

```python blur_with_oak.py -k```

In this mode, users are allowed to turn on/off face blurring by pressing the spacebar.

## Video Demonstration

<video src='./demo.mp4' width=360/>
