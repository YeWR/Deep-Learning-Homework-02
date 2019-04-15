import os
import os.path as osp
import argparse
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import network
import lr_schedule
from logger import Logger
import pre_process as prep
from data_list import ImageList, make_dset_list


def validate(model, loader):
    net = model['net']
    net.eval()

    acc = 0
    total = 0

    with torch.no_grad():
        for img, label, path in loader:
            ## cuda
            img = img.cuda()
            label = label.cuda()

            feature, output = net(img)
            predict = torch.argmax(output, dim=1)
            pre = np.array([1 if l == pre else 0 for l, pre in zip(label, predict)])

            acc += np.sum(pre)
            total += len(pre)
    acc /= total
    return acc


def train(args):

    ## init logger
    logger = Logger(ckpt_path=args.ckpt_path, tsbd_path=args.vis_path)

    ## pre process
    train_transforms = prep.image_train(augmentation=args.augmentation)
    valid_transforms = prep.image_test()

    train_dset = ImageList(open(args.train_list).readlines(), datadir='', transform=train_transforms)
    valid_dset = ImageList(open(args.valid_list).readlines(), datadir='', transform=valid_transforms)

    train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
    valid_loader = DataLoader(valid_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    ## set the model
    net = None
    if args.net == 'MyOwn':
        ## model C
        net = network.ResNet(network.Bottleneck, [3,4,6,3], weight_init=args.weight_init, use_bottleneck=args.bottleneck, num_classes=args.class_num, weight=args.weight)
    else:
        ## model A -> Resnet50 == pretrained
        ## model B -> Resnet50 == not pretrained
        net = network.ResNetFc(resnet_name=args.net, pretrained=args.pretrained, weight_init=args.weight_init, use_bottleneck=args.bottleneck, new_cls=True, class_num=args.class_num)

    net = net.cuda()
    parameter_list = net.get_parameters()

    ## set optimizer and learning scheduler
    if args.opt_type == 'SGD':
        optimizer = optim.SGD(parameter_list, lr=1.0, momentum=args.momentum, weight_decay=0.0005, nesterov=True)
    elif args.opt_type == 'Adam':
        optimizer = optim.Adam(parameter_list, lr=args.lr, weight_decay=0.0005)
    lr_param = {'lr': args.lr, "gamma": 0.001, "power": 0.75}
    lr_scheduler = lr_schedule.inv_lr_scheduler


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
    total_epochs = 1000
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=total_epochs * len(train_loader))

    ## begin train
    it = 0
    for epoch in range(total_epochs):
        for img, label, path in train_loader:
            ## update log
            it += 1
            logger.step(1)
            total_progress_bar.update(1)

            ## validate
            if it % args.test_interval == 1:
                ## validate
                acc = validate(model, valid_loader)

                ## utils
                logger.add_scalar('accuracy', acc * 100)
                logger.save_ckpt(state={
                    'net': net.state_dict()
                }, cur_metric_val=acc)
                log_str = "iter: {:05d}, precision: {:.5f}".format(it, acc)
                args.log.write(log_str + '\n')
                args.log.flush()

            ## train the model
            net.train(True)
            optimizer = lr_scheduler(optimizer, it, **lr_param)
            optimizer.zero_grad()

            ## cuda
            img = img.cuda()
            label = label.cuda()

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
    parser.add_argument('--pretrained', type=int, default=0, help="the backbone is pretrained")
    parser.add_argument('--bottleneck', type=int, default=0, help="use bottleneck in Resnet")
    parser.add_argument('--augmentation', type=int, default=0, help="the backbone is pretrained")
    parser.add_argument('--weight_init', type=int, default=0, help="init weight in the new layers")
    parser.add_argument('--max_iter', type=int, default=1000, help="tradeoff of residual block")
    parser.add_argument('--weight', type=float, default=1., help="tradeoff of residual block")
    parser.add_argument('--data_dir', type=str, default='../data/', help="The data set directory")
    parser.add_argument('--class_num', type=int, default=65, help="class number of the task")
    parser.add_argument('--batch_size', type=int, default=64, help="class number of the task")
    parser.add_argument('--test_interval', type=int, default=200, help="interval of two continuous test phase")
    parser.add_argument('--opt_type', type=str, default='SGD', help="the optimization type: SGD or Adam")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum for SGD")
    parser.add_argument('--lr', type=float, default=0.001, help="initial learning rate")
    parser.add_argument('--debug_str', type=str, default='')

    args = parser.parse_args()

    ## prepare the path for models and visualizations
    output_str = "pretrained-%s,bottleneck-%s,augmentation-%s,weight_init-%s,opt_type-%s,momentum-%s,lr-%s,batch-%s,weight-%s,debug-%s" % (args.pretrained, args.bottleneck, args.augmentation, args.weight_init, args.opt_type, args.momentum, args.lr, args.batch_size, args.weight, args.debug_str)

    args.vis_path = "vis/" + output_str
    args.ckpt_path = "models/" + output_str

    args.train_list = osp.join(args.data_dir, 'train.txt')
    args.valid_list = osp.join(args.data_dir, 'valid.txt')

    if not osp.exists(args.train_list):
        make_dset_list(filename=args.train_list, path=osp.join(args.data_dir, 'train'))
    if not osp.exists(args.valid_list):
        make_dset_list(filename=args.valid_list, path=osp.join(args.data_dir, 'valid'))

    if not osp.exists(args.ckpt_path):
        os.system('mkdir -p ' + args.ckpt_path)
    args.log = open(osp.join(args.ckpt_path, "log.md"), "w")
    if not osp.exists(args.ckpt_path):
        os.mkdir(args.ckpt_path)
    if not osp.exists('vis'):
        os.mkdir('vis')

    print('Config: \n' + str(args) + '\n')
    args.log.write('Config: \n' + str(args) + '\n')
    args.log.flush()

    ## train the model
    train(args)
