from pathlib import Path
import cv2
import depthai as dai
import numpy as np
import time
from sort import *
import argparse

class FPSHandler():
    def __init__(self):
        self.timestamp = time.time()
        self.start = time.time()
        self.frame_cnt = 0

    def next_iter(self):
        self.timestamp = time.time()
        self.frame_cnt += 1

    @property
    def fps(self):
        dt = self.timestamp - self.start
        if dt == 0:
            print("[Warning] division by zero during fps calculation")
            return 0
        return self.frame_cnt / dt
    

class OAKBlur():
    def __init__(self, blobPath, fps, img_size, conf, keyboard):
        if keyboard:
            try:
                import keyboard as KB
                KB.on_press(self.switch_key) # start the keyboard listener
            except:
                print("Unable to import keyboard module")
        
        # Get weights (i.e. *.blob files)
        nnPath = str((Path(blobPath)).resolve().absolute())
        if not Path(nnPath).exists():
            raise FileNotFoundError(f'Required *.blob file/s not found"')
        
        # Create pipeline
        self.device = dai.Device()
        self.pipeline = dai.Pipeline()

        # Define sources and outputs
        self.camRgb = self.pipeline.create(dai.node.ColorCamera)
        self.inputManip = self.pipeline.create(dai.node.ImageManip)
        self.detectionNetwork = self.pipeline.create(dai.node.YoloDetectionNetwork)
        self.xoutRgb = self.pipeline.create(dai.node.XLinkOut)
        self.xoutRgb.setStreamName("rgb")
        self.nnOut = self.pipeline.create(dai.node.XLinkOut)
        self.nnOut.setStreamName("nn")
        
        # camRgb settings
        self.pre_size = 860, 540
        self.camRgb.setPreviewSize(self.pre_size[0], self.pre_size[1])
        self.camRgb.setIspScale(1, 2)
        self.camRgb.setVideoSize(self.pre_size[0], self.pre_size[1])
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(fps)
        self.camRgb.setPreviewKeepAspectRatio(False)
        self.camRgb.setPreviewNumFramesPool(2)
        self.camRgb.setIspNumFramesPool(1)
        self.camRgb.setVideoNumFramesPool(1)
        self.camRgb.setStillNumFramesPool(0)
        try:
            self.calibData = self.device.readCalibration2()
            focalLength = self.calibData.getLensPosition(dai.CameraBoardSocket.RGB)
            if focalLength:
                self.camRgb.initialControl.setManualFocus(focalLength)
        except:
            raise

        # inputManip (resizing) settings
        self.inputManip.initialConfig.setResize(img_size, img_size)
        self.inputManip.initialConfig.setKeepAspectRatio(False) 
        self.inputManip.setNumFramesPool(1)

        # detectionNetwork settings
        self.detectionNetwork.setConfidenceThreshold(conf)
        self.detectionNetwork.setNumClasses(1)
        self.detectionNetwork.setCoordinateSize(4)
        # uncomment these lines for anchor-based models (e.g. yolov7tiny)
        # self.detectionNetwork.setAnchors([12.0, 16.0, 19.0, 36.0, 40.0, 28.0, 36.0, 75.0, 76.0, 55.0, 72.0, 146.0, 142.0, 110.0, 192.0, 243.0, 459.0, 401.0])
        # sides = ["side" + str(img_size//8), "side" + str(img_size//16), "side" + str(img_size//32)]
        # self.detectionNetwork.setAnchorMasks({sides[0]: [0, 1, 2], sides[1]: [3, 4, 5], sides[2]: [6, 7, 8]}) 
        self.detectionNetwork.setIouThreshold(0.5)
        self.detectionNetwork.setBlobPath(nnPath)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)
        self.detectionNetwork.input.setQueueSize(1)

        # Linking
        self.camRgb.preview.link(self.xoutRgb.input)
        self.camRgb.preview.link(self.inputManip.inputImage)

        self.inputManip.out.link(self.detectionNetwork.input)
        self.detectionNetwork.out.link(self.nnOut.input)

        self.device.startPipeline(self.pipeline)

        # Initialization
        self.sort_tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.5) 
        self.tracked_dets = None
        self.qRgb = self.device.getOutputQueue(name="rgb", maxSize=2, blocking=False)
        self.qDet = self.device.getOutputQueue(name="nn", maxSize=2, blocking=False)
        self.blurring = True
        self.fps_handler = FPSHandler()

    def get_pipeline(self):
        return self.pipeline
    
    """Function to read events to turn on / off blurring by pressing the space"""
    def switch_key(self, event):
        if event.name == 'space':
            self.blurring = not self.blurring
            if self.blurring:
                print("Blurring is activated.")
            else:
                print("Blurring is deactivated.")

        return True # Return true to continue the keyboard listener
    
    """Function to blur the detected faces"""
    def blur_faces(self, img, bbox, ksize = (60, 60)): 
        for box in bbox:
            x1, y1, x2, y2 = [int(i) for i in box]
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, self.pre_size[0]), min(y2, self.pre_size[1])
            if self.blurring:
                img[y1:y2, x1:x2] = cv2.boxFilter(img[y1:y2, x1:x2], -1, ksize)
        return img
    
    def infer(self):
        frame = self.qRgb.get().getCvFrame()
        dets_to_sort, bbox_xyxy, detections = np.empty((0, 6)), None, []

        inDet = self.qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections

        self.fps_handler.next_iter()
        cv2.putText(frame, "NN fps: {:.2f}".format(self.fps_handler.fps),
                    (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255))
        
        for detection in detections:
            dets_to_sort = np.vstack((dets_to_sort, 
                        np.array([self.pre_size[0]*detection.xmin, self.pre_size[1]*detection.ymin, self.pre_size[0]*detection.xmax, self.pre_size[1]*detection.ymax, detection.confidence, detection.label])))
        tracked_dets = self.sort_tracker.update(dets_to_sort, unique_color=True)

        if len(tracked_dets) > 0:
            bbox_xyxy = tracked_dets[:,:4]
            frame = self.blur_faces(frame, bbox_xyxy)
            
        cv2.imshow('Blur with OAK', frame)

        return frame
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specification of model path, fps, resolution, etc.')
    parser.add_argument('-b', '--blobPath', action = 'store', default = './yolov8n-face_openvino_2022.1_6shave.blob', help = 'specify path of *.blob file')
    parser.add_argument('-f', '--fps', action = 'store', default = 35, help = "specify fps, maximum capped at 35")
    parser.add_argument('--img_size', action = 'store', default = 256, help = "specify size of image to be send to the detector, default is 256")
    parser.add_argument('-c', '--conf', action = 'store', default = 0.2, help = "setup confidence threshold, default is 0.2")
    parser.add_argument('-k', '--keyboardOn', action = 'store_true', help = "turn on keyboard control (by pressing the space bar) of blurring or not")

    args = parser.parse_args()
    oak = OAKBlur(blobPath = args.blobPath, fps = int(args.fps), img_size = int(args.img_size), 
                  conf = float(args.conf), keyboard = args.keyboardOn)

    while True:
        oak.infer()
        if cv2.waitKey(1) == ord('q'):
            break
