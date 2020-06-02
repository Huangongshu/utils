# -*- coding: utf-8 -*-
#author: huan

import argparse
import models
from utils import *
import os
import pickle
import sys

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    
    #train
    parse.add_argument('--i_e', default = 1,
                       help = 'the initial epoch of the train', type = int)
    parse.add_argument('--e_n', default = 1000,
                       help = 'the epoch number of the train', type = int)    
    parse.add_argument('--lr', default = 1e-4, type = float, help = 'the learning rate')
    parse.add_argument('--batch_sz', default = 40,
                       help = 'the batch size', type = int)  
    parse.add_argument('--project', default = '0',
                       help = 'the test project name', type = str)  

    parse.add_argument('--train_w', default = 10, 
                       help = 'the multiprocessing number or the threading number', type = int)  


    #path
    parse.add_argument('--output', default = './output',
                       help = 'the result output path of the train', type = str)  
    parse.add_argument('--config_out', default = './config',
                       help = 'the config output path of the train', type = str)
    parse.add_argument('--model_path', default = './model',
                       help = 'save model in the path', type = str)
    parse.add_argument('--test_output', default = './result',
                       help = 'the test result output path', type = str)
    
    parse.add_argument('--root', default = './chromosome',
                       help = 'the image base path', type = str)    
    parse.add_argument('--train_info_path', default = './chromosome/chromosome_info.txt')  
    parse.add_argument('--eval_info_path', default = './chromosome/chromosome_info.txt')                
    parse.add_argument('--test_path', default = './chromosome/D00122_01-01_2001140911330___1___0.jpg', type = str)

    parse.add_argument('--weights', default = 'efficientnet-b0_imagenet_1000_notop.h5',
                       help = 'the base model weights path', type = str)    
    parse.add_argument('--get_model_path', default = 'test_model.h5',
                       help = 'Test model save path', type = str) 

    #model
    parse.add_argument('--i_h' , default = 224,
                       help = 'the image hight', type = int)
    parse.add_argument('--i_w' , default = 224,
                       help = 'the image weight', type = int)   
    parse.add_argument('--channel' , default = 3,
                       help = 'the image channel', type = int)     
    parse.add_argument('--m', default = 'efficientnetb0',
                       help = 'the used model name', type = str)

    parse.add_argument('--classnum' , default = 2,
                       help = 'the image hight', type = int)
    parse.add_argument('--optim', default = 'adam',
                       help = 'the used model name', type = str)
    parse.add_argument('--data_mode', default = 'categorical',
                       help = 'the class mode', type = str)
    parse.add_argument('--head', default = 24, 
                       help = 'the head number', type = int)

    parse.add_argument('--DP', default = 0.5,
                       help = 'the used model name', type = int)
    parse.add_argument('--h_u', default = 50,
                       help = 'the used model name', type = int)


    #control
    parse.add_argument('--bal', default = True,
                       help = 'balance sample in the model training', dest = 'balance', action = 'store_false') 
    parse.add_argument('--q_down', default = True, 
                       help = 'Drop the sample down', action = 'store_true')  
    parse.add_argument('--reg', default = 0.0, 
                       help = 'regularizer', type = float)    

    #argument
    parse.add_argument('--flip', default = True,
                       help = 'To control whether the data is flipping')                      
    parse.add_argument('--rotation', default = 180,
                       help = 'used to rotate the data', type = int) 
    parse.add_argument('--shift', default = 0.2,
                       help = 'control the image shift range', type = int)     
    parse.add_argument('--bright', default = 0.1,
                       help = 'control image brightness', type = int) 

    parse.add_argument('--shear', default = 0.2,
                       help = 'control  the image shear range', type = int)     
    parse.add_argument('--zoom', default = 0.1,
                       help = 'control the image zoom range', type = int) 

    #use to control the model training
    parse.add_argument('--is_train', default = True,
                       help = 'the run mode, i.e. train or test', action = 'store_false')   
    parse.add_argument('--test_epoch', default = 0,
                       help = 'restart in one epoch,i.e. 23 epoch', type = int) 
    parse.add_argument('--restart', default = False,
                       help = 'restart train or test in a trained model', action = 'store_true')     
    parse.add_argument('--save_fre', default = 1, 
                       help = 'Frequency of save the model', type = int)  

    parse.add_argument('--th', default = 0.5,
                       help = 'the threshold value', type = float)      
    parse.add_argument('--split', default = 0.0, 
                       help = 'split the data coefficient', type = float)
    parse.add_argument('--get_model', default = False, 
                       help = 'get the finally model', action = 'store_true')

    args = parse.parse_args()

    args.laster = args.i_e
    args.loss = []
    args.acc = []
    args.test_acc = []

    args.mean_sen = []
    args.mean_spe = []
    args.mean_f1 = []  
    args.input_sz = (args.i_h, args.i_w, args.channel)  

    args.best_test_acc = [0, 0]
    args.best_test_sen = [0, 0]
    args.best_test_spe = [0, 0]    
    args.best_test_f1 = [0, 0]  

    M = models.__dict__[args.m](args)
    model = M.set_para(args)    
    if args.restart:
        reg = args.reg
        test_epoch = args.test_epoch
        th = args.th
        reg = args.reg
        save_fre = args.save_fre

        is_train = args.is_train
        test_path = args.test_path
        root = args.root    
        get_model = args.get_model

        with open(os.path.join(args.config_out, args.project, "config.pickle"), 'rb') as f:
            config_args = pickle.load(f)  
            
        args = config_args
        args.save_fre = save_fre
        args.test_path = test_path
        args.is_train = is_train
        args.reg = reg

        args.th = th
        args.root = root
        args.reg = reg
        args.get_model = get_model

        if test_epoch:
            args.laster = test_epoch          
        model.load_weights('{}/{}/{}_{}.h5'.format(args.model_path, args.project,
                             'model', str(args.laster)), by_name = True)            

        args.i_e = args.laster + 1     
    if args.get_model:
        model.save(args.get_model_path)
        print('get a model')
        sys.exit(0)

    # make dir  
    check_path(os.path.join(args.model_path, args.project))
    check_path(os.path.join(args.config_out, args.project))
    check_path(os.path.join(args.output, args.project))    
    check_path(os.path.join(args.test_output, args.project))

    if args.is_train:
        #data  
        if args.split:
            x_train, y_train, x_eval, y_eval = load_data(args.train_info_path, args)
        else:
            x_train, y_train = load_data(args.train_info_path, args)
            x_eval, y_eval = load_data(args.eval_info_path, args)     
        trainset = DataSet(args, x_train, y_train, 'trainfilename')
        evalset = DataSet(args, x_eval, y_eval, 'evalfilename')   

        for epoch in range(args.i_e, args.e_n):
            args.epoch = epoch
            print('start %s epoch' % epoch)

            #train
            model, args = train(model, trainset, args)
            #eval    
            model, args = evaluate(model, evalset, args)  
            #save 
            args = save_result(model, args)
    else:
        if 'jpg' in args.test_path or 'png' in args.test_path or \
                    'jpeg' in args.test_path:
            result = test(model, args)
            print('test image result:', result)
        else:
            args.split = 0
            x_test, y_test = load_data(args.test_path, args)         
            testset = DataSet(args, x_test, y_test, 'testfilename')           
            model = evaluate(model, testset, args)
            print('complete test')
