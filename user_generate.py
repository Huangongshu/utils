# -*- coding: utf-8 -*-
#@author huan

from keras.utils import Sequence, to_categorical
import pandas as pd
import numpy as np
import cv2
from multiprocessing import Pool

def _get_data(im, preproce_fun, state):
    if preproce_fun != None:
        if state == 'train':
            imx, imgx = preproce_fun(im)
            return imx, imgx
        else:
            return preproce_fun(im)
    return im

class Gen(Sequence):
    '''
    mode: binary: [0, 1, 0...]
          categorical:[[1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0],
                      [1, 0, 0, 0, 0]..]
          None: only return the image
    '''    
    def __init__(self, x, y, batch_size,
                 preprocess_fun = None,
                 worker = 10,
                 balance = True,
                 shuffle = True,
                 mode = 'binary',
                 sample_ratio = None,
                 shuffle_num = 5,
                 state = 'train'):
        '''
        sample_ratio: its determines the proportion of categories.
        mode: its determines the categories.e.g. binary, categorical and None 
        shuffle: if true, shuffle the data at the end of each epoch
        '''
        self.mode = mode
        self.state = state #control the output type of the data
        self.sample_ratio = sample_ratio #control the class ratio in the batch, its need that the balance is True
        self.balance = balance #control the class balance
        self.shuffle_num = shuffle_num
        self._x = x
        self._y = y

        self.shuffle = shuffle          
        self._classes = np.unique(self._y)
        self._num_classes = len(self._classes)
        self.batch_size = batch_size
        if self.balance:
            self.batch_size = int(self.batch_size / self._num_classes)
        self.preprocess_fun = preprocess_fun
        self.worker = worker  
        if self.state == 'train': 
            self._shuffle_data(self.shuffle_num)

    def __len__(self):
        if self.balance:
            max_len = 0
            for i in self._classes:
                length = len(np.where(self._y == i)[0])
                if isinstance(self.sample_ratio, tuple) or isinstance(self.sample_ratio, list):
                    length = length / self.sample_ratio[i] #classes 0 batch size is 6 ,other is 2
                if length > max_len: 
                    max_len = length   
            return int(np.ceil(max_len / self.batch_size))        
        else:
            return int(len(self._x) / self.batch_size)

    def __getitem__(self, idx):
        #get the index of the data
        index_list = []
        if self.balance:       
            for i in self._classes: 
                index = list(np.where(self._y == i)[0])
                if isinstance(self.sample_ratio, tuple) or isinstance(self.sample_ratio, list):
                    max_len = np.ceil(len(index) / self.batch_size * self.sample_ratio[i])#classes 0 batch size is 6 ,other is 2
                else:
                    max_len = np.ceil(len(index) / self.batch_size)    

                if  max_len < (idx + 1) * self.batch_size:
                    idx = idx % max_len                
                    s = slice(int(idx * self.batch_size),
                              int((idx + 1) * self.batch_size))
                    index = index[s]
                    index_list.append(index)  
                else:
                    s = slice(int(idx * self.batch_size),
                              int((idx + 1) * self.batch_size))
                    index = index[s]
                    index_list.append(index) 
            index_list = np.concatenate(index_list)         
                               
        else:
            if (idx + 1) * self.batch_size < len(self._y):
                index_list = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size)  
            else:
                index_list = np.arange(idx * self.batch_size, len(self._y))  
        np.random.shuffle(index_list) # shuffle on the batch  end    

        #get the data by the index
        if self.mode == 'binary':
            return self._augments(self._x[index_list], self.preprocess_fun),\
                           self._y[index_list]   

        elif self.mode == 'categorical':
            if self.state == 'train':
                x, gx = self._augments(self._x[index_list], self.preprocess_fun)
                return {'input1': x, 'input2': gx}, None
            else:
                return  self._augments(self._x[index_list], self.preprocess_fun),
                         np.asarray(to_categorical(self._y[index_list], num_classes = self._num_classes))                
        elif self.mode == 'None':
            return self._augments(self._x[index_list], self.preprocess_fun) 
        else:
            raise TypeError('unknown classes mode', self.mode)
        
    def _augments(self, x, fun = None):
        if  self.worker: 
            pool = Pool(self.worker)
            result = [pool.apply_async(_get_data, args = (im, fun, self.state)) for im in x]
            pool.close()
            pool.join()
            collection = [re.get() for re in result]
            return np.asarray(collection)
        else:
            collection_x = []
            collection_gx = []
            for im in x:
                imx, imgx = _get_data(im, fun, self.state)
                collection_x.append(imx)
                collection_gx.append(imgx)
            return  np.asarray(collection_x), np.asarray(collection_gx)

    def _shuffle_data(self, shuffle_num):
        index = np.arange(len(self._y))
        for i in range(shuffle_num):
            np.random.shuffle(index)
        self._x = self._x[index]
        self._y = self._y[index] 

    def on_epoch_end(self):
        if self.shuffle:
            self._shuffle_data(10)
            
if __name__ == '__main__':     
    data_path = 'C:/Users/huan/Desktop/data.csv'
    df= pd.read_csv(data_path)
        
    gen = Gen(df, 6, preprocess_fun = preprocess_im_fun, mode = 'jj')
    i = 0
    for data in gen:
        # print('\n',data[1])
        # cv2.imshow(str(i), data[0][0])
        i += 1