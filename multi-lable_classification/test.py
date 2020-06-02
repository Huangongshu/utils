# -*- coding: utf-8 -*-
#author: huan

import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import time

if __name__ == '__main__':
    import efficientnet.tfkeras
    from tensorflow.keras.models import load_model    
    test_path = 'test_model.h5'

    model = load_model(test_path) 
    start = time.time() 
    for i in range(50):
        im = load_img(os.path.join(args.root, args.test_path))
        im = img_to_array(im)
        im = cv2.resize(im, args.input_sz[:2])
        im = np.expand_dims(im, axis=0)
        im = (im - im.mean()) / (im.std() + 1e-7)
        result = model.predict(im)
    end = time.time()
    print((end - start) / 50)
