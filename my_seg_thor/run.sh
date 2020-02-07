#!/bin/bash

# 参数详解
#'--model_name',		 default='DenseUNet161',	 help='model_name'
#'--epochs',	default=120,	 type=int,	metavar='N', help='number of max epochs to run'
#'--start-epoch',	 default=1, type=int,	metavar='N', help='manual epoch number (useful on restarts)'
#'--batch-size',	default=16, type=int, metavar='N', help='mini-batch size (default: 16)'
#'--lr', 	 '--learning-rate',   default=0.01, type=float, metavar='LR',  help='initial learning rate'
#'--momentum',  	default=0.9,  type=float,  metavar='M',  help='momentum'
# '--resume', 	 default='',  type=str, metavar='PATH', help='path to latest checkpoint (default: none)'
#'--weight-decay',	 default=0.00001,	type=float, metavar='W', help='weight decay (default: 1e-5)'
#'--save_dir',	default='SavePath/test/',  type=str,  metavar='SAVE', help='directory to save checkpoint (default: none)'
# '--gpu', 	default='all',  type=str,  metavar='N',   help='use gpu'
#'--patient',	 default=10,  type=int, metavar='N', help='the flat to stop training'
# '--untest_epoch',	 default=10, type=int, metavar='N', help='number of untest_epoch, do not test for n epoch. just for saving time'
#--loss_name',	default='CombinedLoss', type=str, metavar='N', help='the name of loss function'
#'--data_path',	default='../data/data_npy/',	type=str, metavar='N', help='data path'
#'--test_flag',	default=0, type=int, metavar='0, 1, 2, 3', help='the test flag range in 0..9, 10..19, 20..29, 30..39 !'
#'--n_class',		default=5, type=int, metavar='n_class', help='number of classes'
#'--if_dependent', 	default=1,	type=int, metavar='1(True) or 0(False)', help='the flag to use WMCE'
# '--if_closs',	default=1,	type=int, metavar='1(True) or 0(False)', help='if using multi-task learning'

# 训练时，复制以下命令在main.py所在文件夹下终端运行
# python main.py -b 16 --gpu 0,1,2,3 --model_name ResNetC1 --save_dir SavePath/SM --lr 0.01 --if_dependent 0 --if_closs 0 --test_flag 0
