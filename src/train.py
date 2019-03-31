import os
import os.path as osp
import argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from logger import Logger
import pre_process as prep
import network
from data_list import ImageList, make_dset_list

def validate(args):
    pass

def train(args):

    ## init logger
    logger = Logger(ckpt_path=args.ckpt_path, tsbd_path=args.vis_path)

    ## pre process
    train_transforms = prep.image_train(augmentation=args.augmentation)
    valid_transforms = prep.image_test()

    train_dset = ImageList(open(args.train_list).readlines(), datadir=args.data_dir, transform=train_transforms)
    valid_dset = ImageList(open(args.valid_list).readlines(), datadir=args.data_dir, transform=valid_transforms)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    valid_loader = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    ## set the model
    net = None
    if args.net == 'MyOwn':
        ## TODO: model C
        pass
    else:
        ## model A -> Resnet50 == pretrained == not weight init == new classifier
        ## model B -> Resnet50 == not pretrained == not weight init == new classifier
        net = network.ResNetFc(resnet_name=args.net, pretrained=args.pretrained, weight_init=args.weight_init, new_cls=True, class_num=args.class_num)

    net = net.cuda()
    parameter_list = net.get_parameters()

    ## set optimizer
    ## TODO: set optimizer for SGD and Adam
    optimizer = None

    ## gpu
    gpus = args.gpu_id.split(',')
    if len(gpus) > 0:
        print('gpus: ', [int(i) for i in gpus])
        net = nn.DataParallel(net, device_ids=[int(i) for i in gpus])

    ## for save model
    model = {}
    model['net'] = net

    ## log
    logger.reset()

    ## progress bar
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=100000)

    ## begin train
    epoch = 0
    for img, label, path in train_loader:
        ## update log
        logger.step(1)
        total_progress_bar.update(1)
        epoch += 1

        ## validate
        if epoch % args.test_interval == 1:
            ## TODO: validate
            validate(args)

        ## train the model
        net.train(True)
        optimizer.zero_grad()

        feature, output = net(img)

        loss = nn.CrossEntropyLoss()(output, label)

        loss.backward()
        optimizer.step()

        ## vis
        logger.add_scalar('loss', loss)

if __name__=="__main__":
    ## parameters in the training step
    parser = argparse.ArgumentParser(description='Deep Learning homework 2')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--pretrained', type=bool, default=False, help="the backbone is pretrained")
    parser.add_argument('--augmentation', type=bool, default=False, help="the backbone is pretrained")
    parser.add_argument('--data_dir', type=str, default='../data/', help="The data set directory")
    parser.add_argument('--class_num', type=int, default=65, help="class number of the task")
    parser.add_argument('--batch_size', type=int, default=64, help="class number of the task")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--opt_type', type=str, default='SGD', help="the optimization type: SGD or Adam")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD")
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--debug_str', type=str, default='')

    args = parser.parse_args()

    ## prepare the path for models and visualizations
    output_str = "pretrained-%s,augmentation-%s,opt_type-%s,momentum-%s,lr-%s,batch-%s,debug-%s" % (args.pretrained, args.augmentation, args.opt_type, args.momentum, args.lr, args.batch_size, args.debug_str)

    args.vis_path = "vis/" + output_str
    args.ckpt_path = "models/" + output_str

    args.train_list = osp.join(args.data_dir, 'train.txt')
    args.valid_list = osp.join(args.data_dir, 'valid.txt')

    if not osp.exists(args.train_list):
        make_dset_list(filename='train.txt', path=osp.join(args.data_dir, 'train'))
    if not osp.exists(args.valid_list):
        make_dset_list(filename='valid.txt', path=osp.join(args.data_dir, 'valid'))

    if not osp.exists(args.ckpt_path):
        os.system('mkdir -p ' + args.ckpt_path)
    args.log = open(osp.join(args.ckpt_path, "log.txt"), "w")
    if not osp.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    if not osp.exists('vis'):
        os.mkdir('vis')

    args.log.write('Config: \n' + str(args))
    args.log.flush()

    ## train the model
    train(args)
