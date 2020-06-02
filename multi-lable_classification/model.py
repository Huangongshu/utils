# -*- coding: utf-8 -*-
#author: huan

import os
import sys
sys.path.append('./efficientnet-master/')
try:
    import efficientnet.keras as efn
except:
    os.system('python setup install')
    
import keras.layers as kl
import keras.optimizers as ko
import keras.models as km
import keras.regularizers as kr
from keras.applications.resnet50 import ResNet50

_model_name = {'efficientnetb0': efn.EfficientNetB0, 'efficientnetb1': efn.EfficientNetB1,
               'efficientnetb2': efn.EfficientNetB2, 'efficientnetb3': efn.EfficientNetB3,
               'efficientnetb4': efn.EfficientNetB4, 'efficientnetb5': efn.EfficientNetB5,                       
               'efficientnetb6': efn.EfficientNetB6, 'efficientnetb7': efn.EfficientNetB7,
               'resnet': ResNet50}

__all__ = ['efficientnetb0', 'efficientnetb1',
           'efficientnetb2', 'efficientnetb3',
           'efficientnetb4', 'efficientnetb5',
           'efficientnetb6', 'efficientnetb7',
           'resnet']

_change_effi_dc = {'block2a_dwconv': 6, 'block3a_dwconv': 1,
                   'block4a_dwconv': 1, 'block6a_dwconv': 1}

_change_effi_co = {'stem_conv': 48}

_change_resnet = {'ZeroPadding2D':       kl.ZeroPadding2D(padding = (0, 0),
                                                   name = 'conv1_pad'),
                  'pool1_pool':          kl.MaxPooling2D(1,
                                                   name = 'pool1_pool'),
                  'conv3_block1_1_conv': kl.Conv2D(filters = 128,
                                                   kernel_size = (3, 3),
                                                   strides = (1, 1),
                                                   padding = 'same'),
                  'conv4_block1_1_conv': kl.Conv2D(filters = 256,
                                                   kernel_size = (3, 3),
                                                   strides = (1, 1),
                                                   padding = 'same'),
                  'conv5_block1_1_conv': kl.Conv2D(filters = 512,
                                                   kernel_size = (3, 3),
                                                   strides = (1, 1),
                                                   padding = 'same')}

class BaseModel():
    #used to defined the base model
    def __init__(self, config):
        self.q_down = config.q_down
        self.classnum = config.classnum
        self.input_shape = config.input_sz
        self.weights = config.weights 

        self.head = config.head
        self.reg = kr.l2(config.reg) if config.reg else None
        self.config_para = {'Dropout': config.DP, 'h_unit': config.h_u}

        if self.head == 1:
            self._final_act = 'softmax'
            self._final_n = self.classnum
            self.loss = 'categorical_crossentropy'
            
        else:
            self._final_act = 'sigmoid'            
            self._final_n = 1
            self.loss = ['binary_crossentropy'] * self.head
                     
    def _build(self):
        self._base_m = _model_name.get(self.name)(weights = None,
                                include_top = False,
                                input_shape = self.input_shape)
        self._base_m.load_weights(self.weights, by_name = True)
        self.inp = self._base_m.get_layer(index = 0).input  
        self.out = self._base_m.get_layer(index = -1).output        

    def _head(self, x, name = '0'):
        out = kl.Dropout(self.config_para.get('Dropout'), name = 'do1_'  + name)(x)
        out = kl.BatchNormalization(name = 'bn1_'  + name)(out)
        out = kl.Dense(self.config_para.get('h_unit'), kernel_initializer = 'he_normal',
               kernel_regularizer = self.reg,
               activation = 'relu', name = 'ds1_'  + name)(out)
        out = kl.Dropout(self.config_para.get('Dropout'), name = 'do2_'  + name)(out)
        out = kl.BatchNormalization(name = 'bn2_'  + name)(out)                     
        out = kl.Dense(self._final_n, kernel_initializer = 'he_normal',
                 kernel_regularizer = self.reg,          
                 activation = self._final_act, name = 'ds2_'  + name)(out)
        return out
            
    def _classify(self):
        out = kl.GlobalAveragePooling2D()(self.out)
        self._c_l = [self._head(out, str(i)) for i in range(self.head)]
        self._model = km.Model(inputs = self.inp, outputs = [o for o in self._c_l])

    def set_para(self, config):
        if config.optim == 'adam':
            optim = ko.Adam(config.lr)
        elif config.optim == 'sdg':
            optim = ko.SGD(config.lr)  
        if config.restart:
            with open(os.path.join(config.config_out, config.project, "opt.pickle"), 'rb') as f:
                cfg = pickle.load(f)          
            optim = optim.from_config(cfg)
                    
        self._model.compile(optimizer = optim,
                                  loss = self.loss,
                                  metrics = ['acc']
                                  )    
        return self._model

    def get(self):
        return self._model
    
class _efficientnet(BaseModel):
    #used to define the efficientnet base model
         
    def _build(self):
        super(_efficientnet, self)._build()
        if self.q_down:
            self.__quit_downsample() 

    def __quit_downsample(self): 
        for i in range(len(self._base_m.layers)):
            name = self._base_m.layers[i].name
            co_k = _change_effi_co.get(name)
            cd_k = _change_effi_dc.get(name)
            if co_k:
                self._base_m.layers[i] = kl.Conv2D(co_k,
                           kernel_size = (3, 3),
                           strides=(1, 1),
                           padding = 'same',
                           name = name)
            elif cd_k:
                self._base_m.layers[i] = kl.DepthwiseConv2D(
                           depth_multiplier = cd_k,
                           kernel_size = (3, 3),
                           strides=(1, 1),
                           padding = 'same',
                           name = name)

class efficientnetb0(_efficientnet):
    #used to build the efficientnetb0 model
    def __init__(self, config):
        self.name = 'efficientnetb0'
        super(efficientnetb0, self).__init__(config)   
        self._build()
        self._classify()        
    
class efficientnetb1(_efficientnet):
    #used to build the efficientnetb1 model    
    def __init__(self, config):
        self.name = 'efficientnetb1'
        super(efficientnetb1, self).__init__(config)
        self._build()
        self._classify() 

class efficientnetb2(_efficientnet):
    #used to build the efficientnetb2 model    
    def __init__(self, config):
        self.name = 'efficientnetb2'
        super(efficientnetb2, self).__init__(config)    
        self._build()
        self._classify() 
       
class efficientnetb3(_efficientnet):
    #used to build the efficientnetb3 model    
    def __init__(self, config):
        self.name = 'efficientnetb3'
        super(efficientnetb3, self).__init__(config)
        self._build()
        self._classify() 
         
class efficientnetb4(_efficientnet):
    #used to build the efficientnetb4 model    
    def __init__(self, config):
        self.name = 'efficientnetb4'
        super(efficientnetb4, self).__init__(config)    
        self._build()
        self._classify()  

class efficientnetb5(_efficientnet):
    #used to build the efficientnetb5 model    
    def __init__(self, config):
        self.name = 'efficientnetb5'
        super(efficientnetb5, self).__init__(config)   
        self._build()
        self._classify() 
        
class efficientnetb6(_efficientnet):
    #used to build the efficientnetb6 model    
    def __init__(self, config):
        self.name = 'efficientnetb6'
        super(efficientnetb6, self).__init__(config)    
        self._build()
        self._classify() 
       
class efficientnetb7( _efficientnet):
    #used to build the efficientnetb model    
    def __init__(self, config):
        self.name = 'efficientnetb7'
        super(efficientnetb7, self).__init__(config)         
        self._build()
        self._classify() 
     
class resnet(BaseModel):
    #used to build the resnet model    
    def __init__(self, config):
        self.name = 'resnet'
        super(resnet, self).__init__(config)     
        self._build()
        self._classify()     

    def _build(self):
        super(resnet, self)._build()        
        if self.q_down:
            self.__quit_downsample()         
        
    def __quit_downsample(self): 
        for i in range(len(self._base_m.layers)):
            name = self._base_m.layers[i].name
            _change_l = _change_resnet.get(name)
            if _change_l:
                self._base_m.layers[i] = _change_l        
  
