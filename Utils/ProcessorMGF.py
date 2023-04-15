import numpy as np
import cv2
from ThreadedVideoProcessing.VideoThreaded import VideoProcessor, SequenceProcessor
from Utils.ImageUtils import *



class ProcessorMGF(SequenceProcessor):
    def __init__(self, configArgs, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        self.args = configArgs
        self.initializeMGF()

    def initializeMGF(self):
        first_frame = cv2.imread(self.imageList[self.frameNum])
        first_frame = first_frame.astype(np.float32) / 255.
        images = getImagePyramid(first_frame)
        self.im_prev = images[0].copy()
        self.im_initial, self.t0_rough, self.A = pyramid_dehazing(first_frame, *images[:3], args=self.args)
        self.frameNum += 1
        
    def operationMGF(self, frame, timeT):
        
        frame = frame.astype(np.float32) / 255.
        images = getImagePyramid(frame)
        
        # Temporal Coherence
        diff = np.power((images[0] * 255. - self.im_prev * 255.), 2)
        
        w = np.mean(np.exp(-1 * diff / 100))
        
        t0_prev = self.t0_rough
        self.im_prev = images[0]
        
        if w >= 0.85:
            t0 = fast_gradient(images[0], t0_prev, self.args.N, self.args.NN, self.args)
            t0 = np.maximum(t0, 0.05)
            
            result = np.divide((frame-self.A), np.repeat(t0[:,:,np.newaxis], 3, axis=2)) + self.A
        else:
            im_dv = get_size_image_i(images[2], self.args)
            u, v = im_dv.shape[:2]
            A0 = np.mean(maxk(im_dv, dim=0, k=round(u*v*0.05)))
            A0 = np.minimum(A0, 0.99)
            self.A = (1- w) * A0 + w * self.A
            
            if w < 0.5:
                result, t0_rough, _ = pyramid_dehazing(frame, *images[:3], args=self.args, A=self.A)
            else:
                t0 = fast_gradient(images[0], t0_prev, self.args.N, self.args.NN, self.args)
                t0 = np.maximum(t0, 0.05)
                
                result = np.divide((frame - self.A), np.repeat(t0[:,:,np.newaxis], 3, axis=2)) + self.A
                
            result = (result/np.max(result))
        return result*255, timeT
                
            
        


    