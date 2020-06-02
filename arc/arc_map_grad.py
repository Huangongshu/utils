# -*- coding: utf-8 -*-
#author: huan

import numpy as np
import tensorflow as tf
import os
import json
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
import tensorflow.keras.optimizers as ko
import random

def get_path(data_path):
    path = {'train': [], 'test': [], 'evaluation': []}
    for dirname, _, filenames in os.walk(data_path):
        for filename in filenames:
            if 'train' in dirname:
                path['train'].append('{}/{}'.format(dirname, filename))
            elif 'test' in dirname:
                path['test'].append('{}/{}'.format(dirname, filename))
            elif 'evaluation' in dirname:
                path['evaluation'].append('{}/{}'.format(dirname, filename))
    return path

def load_data(path_coll):
    sample_coll = {}
    for p in path_coll:
        with open(p, 'r') as f:
            sample_coll.update({p.split('/')[-1].split('.')[0]: json.load(f)})
    return  sample_coll

def regress_loss(y_true, y_pred):
    e = y_pred -  y_true
    y_bool = tf.cast(tf.less_equal(tf.abs(e), 1), dtype = tf.float32)
    return tf.reduce_mean(y_bool * 0.5 * tf.square(e) + (1 - y_bool) * tf.abs(e))
             
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred -  y_true))
       
class Fit(kl.Layer):
    
    def __init__(self, pool_len, k_shape, k_init = 'he_normal',
                       b_init = 'zeros', k_regu = None, b_regu = None, **kwargs):
        self.k_shape = k_shape
        self.k_init = k_init
        self.b_init = b_init
        self.k_regu = k_regu
        self.b_regu = b_regu        
        self.pool_len = pool_len
        super(Fit, self).__init__(**kwargs)
        
    def build(self, input_shape): 
        self.method_w1 = self.add_weight(shape = (self.k_shape[2], self.pool_len),
                        initializer = self.k_init,
                        name = 'method_w1',
                        regularizer = self.k_regu)
        
        self.method_b1 = self.add_weight(shape = (1, self.pool_len),
                            initializer = self.b_init,
                            name = 'method_b1',
                            regularizer = self.b_regu)          
 
        self.method_w2 = self.add_weight(shape = (self.k_shape[2], self.pool_len),
                        initializer = self.k_init,
                        name = 'method_w2',
                        regularizer = self.k_regu)
        
        self.method_b2 = self.add_weight(shape = (1, self.pool_len),
                            initializer = self.b_init,
                            name = 'method_b2',
                            regularizer = self.b_regu)          
        
        self.method_k = self.add_weight(shape = self.k_shape,
                        initializer = self.k_init,
                        name = 'method_k',
                        regularizer = self.k_regu)
               
        self.built = True    
        
    def call(self, inputs):
        gap = tf.reduce_mean(inputs, axis = (0, 1, 2), name = '1') 
        gap = tf.expand_dims(gap, axis = 0, name = '2')
                 
        #method
        me1 = tf.matmul(gap, self.method_w1)  #dense handle

        me1 = tf.add(me1, self.method_b1)        
        me1 = tf.nn.tanh(me1)         #activation

        me2 = tf.matmul(gap, self.method_w2)  #dense handle

        me2 = tf.add(me2, self.method_b2)        
        me2 = tf.nn.tanh(me2)         #activation            
                   
        me = tf.reduce_sum([me1, me2], axis = 0)
        method = tf.multiply(me[0], self.method_k) #get the method   
        p_conv_k = tf.reduce_mean(method, axis = -1, keepdims = True) #get the method conv2d kernel    
        return tf.nn.conv2d(inputs, p_conv_k, strides = 1, padding = 'SAME')
 
class ARC_Task():
    
    def __init__(self, pool_len = 3, lr = 1e-4):
        #parameter
        self.pool_len = pool_len
        self.lr = lr
        self.opt1 = ko.Adam(lr)
        self.opt2 = ko.Adam(lr)
        #input layer        
        self.input_im = kl.Input(shape = (None, None, 3))
        self.input_feature = kl.Input(shape = (None, None, 20))
        
        #build model
        self._build()

        
    def _build(self):
        feature = kl.Conv2D(10, kernel_size = (3, 3), activation = 'relu',
                             strides = (1, 1), padding = 'same', name = 'conv1')(self.input_im)
        
        feature = kl.BatchNormalization(name = 'bn1')(feature)
        feature = kl.Conv2D(50, kernel_size = (3, 3), activation = 'relu',
                              strides = (1, 1), padding = 'same', name = 'conv2')(feature)  
        feature = kl.BatchNormalization(name = 'bn2')(feature)  

        out = kl.GlobalAveragePooling2D(name = 'GAP')(feature)

        out = kl.Dense(50, activation = 'relu', name = 'dense1')(out)
        out = kl.Dense(100, activation = 'relu', name = 'dense2')(out)                  
        out_size = kl.Dense(2, name = 'output_size')(out)
        w = tf.stop_gradient(tf.round(out_size[0][1]))
        h = tf.stop_gradient(tf.round(out_size[0][0]))
        
        @tf.custom_gradient
        def resize_im(x, y, z):
            feature = x
            def grad(dy):
                return dy * feature
            return tf.image.resize(feature, (y, z)), grad
        
        resize_feature = resize_im(feature, w, h)
        output = self.fit = Fit(pool_len = self.pool_len, k_shape = (3, 3, 50, 1),\
                                k_init = 'he_normal', b_init = 'zeros', k_regu = None, b_regu = None)(resize_feature)    

        self.size = km.Model(inputs = self.input_im, outputs = out_size)                      
        self.context = km.Model(inputs = self.input_im, outputs = output)   

    def accumulate(self, g, opt, accum_grad, loss_re, trainable_variables, step, batch_size = 64, skip_variables = {}):
        grad = g.gradient(loss_re, trainable_variables)    
        if len(accum_grad) == 0:
            accum_grad = grad
        else:
            temp = []
            for g, accum_g, v in zip(grad, accum_grad, trainable_variables):
                if not skip_variables.get(v.name, 0):
                    value = tf.add(g, accum_g)
                    temp.append(value)                   
            accum_grad = temp
        
        if step % batch_size == 0:
            accum_grad = [tf.divide(g, batch_size) for g in accum_grad]                                            
            opt.apply_gradients(zip(accum_grad, trainable_variables))      
            accum_grad = []
        return accum_grad

    def train(self, data, batch_size):
        s_step, c_step = 1, 1
        s_loss, c_loss = [], []
        size_acc = []
        s_accum_grad,  c_accum_grad = [], []
        skip_variables = {'dense1/kernel:0': 1, 'dense1/bias:0': 1,
                          'dense2/kernel:0': 1, 'dense2/bias:0': 1,
                          'output_size/kernel:0': 1, 'output_size/bias:0': 1}
        for d in data:
            x_ = np.array(d['input'], dtype = np.float32)
            y_ = np.array(d['output'], dtype = np.float32)       
            x_ = np.stack([x_] * 3, axis = 2)

            with tf.GradientTape() as g1:
                result = self.size(np.expand_dims(x_, axis = 0))
                loss_re = mse(np.expand_dims(np.asarray(y_.shape), axis = 0), result)
            s_accum_grad = self.accumulate(g1, self.opt1, s_accum_grad, loss_re,
                                 self.size.trainable_variables, s_step, batch_size)         
            s_loss.append(loss_re.numpy())

            resize = self.size.predict(np.expand_dims(x_, axis = 0))
            resize = [round(s) for s in resize[0]]
            if np.all(tuple(resize) == y_.shape):
                size_acc.append(1)    
                with tf.GradientTape() as g2:   
                    output = self.context(np.expand_dims(x_, axis = 0))
                    loss_co = mse(np.expand_dims(y_, axis = (0, 1)), output)
                with open('variables.txt', 'w') as f:
                    f.write('{}'.format(self.context.trainable_variables))

                c_accum_grad = self.accumulate(g2, self.opt2, c_accum_grad, loss_co,
                                 self.context.trainable_variables, c_step, 8, skip_variables = skip_variables)
                c_loss.append(loss_co.numpy())
                c_step += 1
            else:
                size_acc.append(0)   
            s_step += 1                
        return np.mean(s_loss), np.mean(c_loss), np.mean(size_acc) 
        
    def test(self, x, y_):
        x_ = np.array(x, dtype = np.float32) 
        x_ = np.stack([x_] * 3, axis = 2)   
        y_ = np.array(y_, dtype = np.float32)         
        resize = self.size.predict(np.expand_dims(x_, axis = 0)) 
        resize = [round(s) for s in resize[0]]     
        if np.all(tuple(resize) == y_.shape):         
            y_pred = self.context.predict(np.expand_dims(x_, axis = 0)) 

            if np.all(y_pred[0, :, :, 0] == y_):
                return 1, 1
            return 1, 0            
        else:
            return 0, 0
                
data_path = './abstraction-and-reasoning-challenge/'
epoch = 300

path = get_path(data_path)

train_sample = load_data(path['train'])
evaluation_sample = load_data(path['evaluation'])
test_sample = load_data(path['test'])
train_sample.update(evaluation_sample)

train_d = [d for k, v in train_sample.items() for d in v['train']]
eval_d = [d for k, v in train_sample.items() for d in v['test']]

train_data = [data for data in train_d]
arg_ids = np.asarray([1 if np.all(np.asarray(x['input']).shape == np.asarray(x['output']).shape) else 0 for x in train_data])
eval_data = [data for data in eval_d]

ids1 = np.where(arg_ids == 1)
ids0 = np.where(arg_ids == 0)
num = abs(len(ids1) - len(ids0))

if len(ids1) > len(ids0):
    range_ids = ids0[0]
else:
    range_ids = ids1[0]

Task = ARC_Task(pool_len = 3, lr = 1e-3)


for e in range(epoch):
    s_score, c_score = [], []
    new_ind = np.random.choice(range_ids, num, replace = False)
    f_train_data = []
    for i in new_ind:
        f_train_data.append(train_data[i])
    for d in train_data:
        f_train_data.append(d)        
        
    random.shuffle(f_train_data)
    s_loss, c_loss, size_score = Task.train(f_train_data, 32)
       
    for data in eval_data:
        s_s, c_s = Task.test(data['input'], data['output'])
        s_score.append(s_s)
        c_score.append(c_s)        

    print('epoch: {} size loss:{:.4f} context loss:{:.4f} size acc: {:.4f}\n'.format(e,
                        s_loss, c_loss, size_score))        
    print('epoch: {} test size score {:.4f} context score {:.4f}\n'.format(e, np.mean(s_score), np.mean(c_score)))
    with open('loss.txt', 'a') as f:
        f.write('epoch: {} size loss: {:.4f} context loss:{:.4f} size acc: {:.4f} test size score: {:.4f} context score {:.4f}\n'.format(e,
                    s_loss, c_loss, size_score, np.mean(s_score), np.mean(c_score)))
