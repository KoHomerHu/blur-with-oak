# Blur with OAK

Real-time face detection + tracking + blurring using OAK-D-Lite. We use the [`yolov8n-face.pt`](https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt) model from the [yolov8-face](https://github.com/akanametov/yolov8-face/tree/dev) repo and the implementation of SORT is adapted from the [haroonshakeel/yolov7-object-tracking](https://github.com/haroonshakeel/yolov7-object-tracking) repo. The script can be run on edge devices (e.g. Raspberry Pi) without a significant drop in FPS (near 35 which is the capped maximum on OAK-D-Lite).

What we are doing is similar to the DepthAI experiment [gen2-blur-faces](https://github.com/luxonis/depthai-experiments/tree/master/gen2-blur-faces), although we use YOLOv8n and the SORT algorithm instead of mobilenet detection and zero term object tracking node on OAK. In other words, this repo may be considered as an "upgrade" to  [gen2-blur-faces](https://github.com/luxonis/depthai-experiments/tree/master/gen2-blur-faces), as we have achieved:
- Faster inference speed (Testing `gen2-face-blur.py` which I modified to calculate and show the FPS, the result was around 30)
- More robust detection and tracking pipeline
- Fully utilizing the specified HFOV of OAK camera

## Argument `keyboardOn`

Run the following command to activate the keyboard switch for activating/deactivating blurring: 

```python blur_with_oak.py -k```

In this mode, users are allowed to turn on/off face blurring by pressing the spacebar.
