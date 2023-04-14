import numpy as np
import cv2
from ThreadedVideoProcessing.VideoThreaded import VideoProcessor
from Utils.ImageUtils import *



class ProcessorMGF(VideoProcessor):
    def __init__(self, configArgs, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        self.args = configArgs
        self.initializeMGF()

    def initializeMGF(self):
        ret, first_frame = self.cap.read()
        if not ret:
            Exception("Initialization unsuccessfull. VideoCapture is unable to read.")
        images = getImagePyramid(first_frame)
        first_frame = first_frame.astype(np.float32) / 255.
        self.im_prev = images[0].copy()
        self.im_initial, self.t0_rough, self.A = pyramid_dehazing(first_frame, *images[:3], args=self.args)
        self.frameCount -= 1
        
    def operationMGF(self, frame, timeT):
        
        images = getImagePyramid(frame)
        frame /= 255.
        
        # Temporal Coherence
        diff = np.power((images[0] * 255.0 - self.im_prev * 255.), 2)
        
        w = np.mean(np.exp(-1 * diff / 100))
        
        t0_prev = self.t0_rough
        self.im_prev = images[0]
        
        if w >= 0.85:
            t0 = fast_gradient(images[0], t0_prev, self.args.N, self.args.NN)
            t0 = np.max(t0, 0.05)
            
            result = np.divide((frame-self.A), t0) + self.A
        else:
            im_dv = get_size_image_i(images[2], self.args)
            u, v = im_dv.shape[:2]
            A0 = np.mean(maxk(im_dv, dim=0, k=round(u*v*0.05)))
            A0 = min(A0, 0.99)
            self.A = (1- w) * A0 + w * self.A
            
            if w < 0.5:
                result, t0_rough, _ = pyramid_dehazing(frame, *images[:3], args=self.args, A=self.A)
            else:
                t0 = fast_gradient(images[0], t0_prev, self.args.N, self.args.NN)
                t0 = np.maximum(t0, 0.05)
                
                result = np.divide((frame - self.A), t0) + self.A
                
        return result, timeT
                
            
        


    