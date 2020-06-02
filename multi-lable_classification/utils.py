# -*- coding: utf-8 -*-
#author: huan

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
from sklearn.metrics import confusion_matrix,f1_score, accuracy_score
import pickle
import cv2
import matplotlib.pyplot as plt
from keras.utils import to_categorical

def check_path(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def load_data(path, config):
    x, y = [], []
    
    with open(path, 'r') as f:
        file = f.readlines()    
        
    for data in file:
        s_list = data.split(',')
        x.append(s_list[0])
        y.append([float(s.strip()) for s in s_list[1:]])
    x = np.asarray(x)
    y = np.asarray(y)     
    if config.split:
        s = int(len(x) * (1 - config.split))
        x_train = x[:s]
        x_test = x[s:]
        y_train = y[:s]
        y_test = y[s:]     
        print('complete the data load')
        return x_train, y_train, x_test, y_test
    else:
        print('complete the data load')
        return x, y  

def save_filename(path, data):
    with open(path, 'w') as f:
        for name in data:
            f.write('{}\n'.format(name))

def best_judge(best_val, val, config):
    if best_val[1] < val[-1]:
        best_val[1] = val[-1]
        best_val[0] = config.epoch 
    return best_val   

def save_result(model, config):
    if config.epoch % config.save_fre == 0: 
        config.laster = config.epoch
        config.best_test_acc = best_judge(config.best_test_acc, config.test_acc, config)
        config.best_test_sen = best_judge(config.best_test_sen, config.mean_sen, config)
        config.best_test_spe = best_judge(config.best_test_spe, config.mean_spe, config)
        config.best_test_f1 = best_judge(config.best_test_f1, config.mean_f1, config)    

        model.save_weights('{}/{}/{}_{}.h5'.format(config.model_path, config.project,
                     'model', str(config.laster)))                                 
        with open(os.path.join(config.config_out, config.project, "config.pickle"), 'wb') as f:
            pickle.dump(config, f)
        with open(os.path.join(config.output, config.project, "best_target.txt"), 'w') as f:
            f.write('epoch: {} acc: {:.4f}sen: {:.4f} spe: {:.4f} fl: {:.4f}'.format(config.best_test_acc[0],
                                                                                     config.best_test_acc[1],
                                                                                     config.best_test_sen[1],
                                                                                     config.best_test_spe[1],
                                                                                     config.best_test_f1[1]))

        print('complete {} model save'.format(config.laster))

    return config
 
def plot(data, subplot = True, name = [], save_path = 'train.jpg'):
    plt.figure(figsize = (8, 16))
    name = name
    if len(name) == 0:
        name = [str(i) for i in range(len(data))]
    c = ['r', 'b', 'y', 'g']
    if subplot:
        for i, da in enumerate(data):
            plt.subplot(len(data), 1, i + 1)
            plt.title(name[i])
            plt.plot(da, label = name[i], color = c[i])
    else:
        plt.figure()
        for da, na in zip(data, name):
            plt.plot(da, label = na)
    plt.legend(name)
    plt.savefig(save_path)
    plt.close()

def evaluate(model, dataset, config):
    if config.classnum == 2:
        average = 'binary'
    else:
        average = None
    test_set = dataset.get_data(False, False)
    y, re = [], [] 
    if config.head == 1:     
        for i in range(len(test_set)):
            y.append(test_set[i][1])
            re.append(model.predict(test_set[i][0]))
        y = np.concatenate(y, 0)
        re = np.concatenate(re, 0)   
        y = np.argmax(y, 1)
        re = np.argmax(re, 1)
        y_true = [y]
        y_pred = [re]        
    else: 
        for i in range(len(test_set)):      
            _y = [to_categorical(o, num_classes = 24) for o in test_set[i][1]]
            temp = _y[0]
            for valu in _y[1:]:
                temp = np.logical_or(valu, temp)
            temp = temp.astype(np.float32)
            y.append(temp)
            y_ = model.predict(test_set[i][0])
            temp = np.where(np.asarray(y_) >= config.th, 1, 0)         
            re.append(temp[:, :, 0])    
        y = np.concatenate(y, 0)
        y_pred = np.concatenate(re, 1)   
        y_pred = np.transpose(y_pred, (1, 0))
        y_true = y

    acc, f1, sen, spe = [], [], [], []
    for h in range(config.head):    
        u = confusion_matrix(y_true[:, h], y_pred[:, h], labels = [0, 1])
        for i in range(len(u)):
            fp = np.sum(u[:, i]) - u[i][i]
            tp = u[i][i]  
            fn = np.sum(u[i], axis = 0) - u[i][i]
            tn = np.sum(u) - tp - fp - fn
            temp1 = tp / (tp + fn)
            temp2 = tn / (tn + fp)
            if config.classnum == 2:
                sen.append(temp2)
                spe.append(temp1)
                break           
            sen.append(temp1)
            spe.append(temp2)

        acc.append(accuracy_score(y_true[:, h], y_pred[:, h]))
        f1.append(f1_score(y_true[:, h], y_pred[:, h], average = average))
  
    if config.test_epoch and config.restart:
        e = config.laster
        output = config.test_output
    else:
        e = config.epoch
        output = config.output

    print('acc:{:.4f} sen:{:.4f} spe:{:.4f} f1:{:.4f}\n'.format(np.mean(acc),
                             np.mean(sen), np.mean(spe), np.mean(f1)))   
    
    with open('{}/{}/{}'.format(output, config.project, 'result.txt'), 'a') as f:
        f.write('epoch: {} tests acc:{:.4f}\n'.format(e,
                            np.mean(acc)))
        for i in range(len(sen)):
            f.write('class {} sen:{:.4f} spe:{:.4f} f1:{:.4f}\n'.format(i, sen[i], spe[i], f1[i]))

    with open('{}/{}/{}'.format(output, config.project, 'sen_spe.txt'), 'a') as f:
        f.write('epoch: {} acc:{:.4f} sen:{:.4f} spe:{:.4f} f1:{:.4f}\n'.format(e,
                np.mean(acc), np.mean(sen), np.mean(spe), np.mean(f1)))    
        config.test_acc.append(np.mean(acc))

    config.test_acc.append(np.mean(acc))
    config.mean_sen.append(np.mean(sen))
    config.mean_spe.append(np.mean(spe))
    config.mean_f1.append(np.mean(f1))

    save_path1 = os.path.join(output, config.project, 'test1.jpg')
    save_path2 = os.path.join(output, config.project, 'test2.jpg')    

    plot([config.test_acc, config.mean_sen, config.mean_spe, config.mean_f1],
                 subplot = True, name = ['test_acc', 'sensitivity', 'specificity', 'f1_score'],
                 save_path = save_path1)
    plot([config.test_acc, config.mean_sen, config.mean_spe, config.mean_f1],
                 subplot = False, name = ['test_acc', 'sensitivity', 'specificity', 'f1_score'],
                 save_path = save_path2)
    return model, config

def train(model, dataset, config):
    trainset = dataset.get_data()   
    if config.head == 1:
        his = model[0].fit_generator(trainset,
                                steps_per_epoch = len(trainset),
                                epochs = 1,
                                max_queue_size = 30,
                                verbose = 0,
                                workers = config.train_w,
                                use_multiprocessing = True,
                                )
        loss = his.history['loss'][-1]
        acc = his.history['acc'][-1]
    else:
        log1 = []
        log2 = []
        for i in range(len(trainset)):
            _x, _y = trainset[i]
            _y = [to_categorical(y, num_classes = 24) for y in _y]
            temp = _y[0]
            for valu in _y:
                temp = np.logical_or(valu, temp)
            temp = temp.astype(np.float32)                
            _y = temp
            val = model.train_on_batch(_x, [_y[:, i] for i in range(config.head)])
            log1.append(np.mean(val[1:25]))
            log2.append(np.mean(val[25:]))

        loss = np.mean(log1)
        acc = np.mean(log2)      

    with open('{}/{}/{}'.format(config.output, config.project, 'loss.txt'), 'a') as f:
        f.write('epoch: {} train loss: {:.4f} accuracy:{:.4f}\n'.format(config.epoch,
                            loss, acc))
        
    print('epoch: {} train loss: {:.4f} accuracy:{:.4f}\n'.format(config.epoch,
                            loss, acc))
    
    config.loss.append(loss)
    config.acc.append(acc) 
    save_path1 = os.path.join(config.output, config.project, 'train1.jpg')
    save_path2 = os.path.join(config.output, config.project, 'train2.jpg')

    plot([config.loss, config.acc], name = ['loss', 'acc'], save_path = save_path1)
    plot([config.loss, config.acc], subplot = False, name = ['loss', 'acc'], save_path = save_path2)
    return model, config

def test(model, config):
    im = preproce_fun(config.test_path)
    return model.predict(im)

class DataSet():
    
    def __init__(self, config, x_train, y_train, name = 'train'):
        self.config = config
        self._directory = config.root
        self._target_size = config.input_sz
        self._batch_size = config.batch_sz
        
        self._x = x_train
        self._y = y_train
        self._head = config.head
        self._sample = ImageDataGenerator(rotation_range = config.rotation,
                                           width_shift_range = config.shift,
                                           height_shift_range = config.shift,
                                           brightness_range = (config.bright, config.bright),
                                           shear_range = config.shear,
                                           zoom_range = config.zoom,
                                           horizontal_flip = config.flip,
                                           vertical_flip = config.flip,
                                           rescale = 1 / 255.,                                     
                                           channel_shift_range = 0.,)
        
        save_filename('{}/{}/{}.txt'.format(config.output, config.project,
                            name), x_train)
    
    def get_data(self, balance = True, shuffle = True):
        x_list, y_list = self._x, self._y           
        y_name = ['class' + str(i) for i in range(len(y_list[0]))]
        y_list = np.asarray(y_list)        
        data = {'filename': x_list}
        for ii, na in enumerate(y_name):
            data.update({na: y_list[:, ii]}) 
        df = pd.DataFrame(data)

        return self._sample.flow_from_dataframe(df, self._directory,
                                             target_size = self._target_size[:2],
                                             y_col = y_name,
                                             class_mode = 'multi_output',
                                             batch_size = self._batch_size, 
                                             shuffle = shuffle
                                             )
        
    def get_label(self):
        return self._y
    
