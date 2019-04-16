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
from data_list import ImageList, make_test_list

def test(args):

    test_transforms = prep.image_test()
    test_dset = ImageList(open(args.test_list).readlines(), datadir='', transform=test_transforms)
    valid_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    ## set the model
    net = None
    if args.net == 'MyOwn':
        ## model C
        net = network.ResNet(network.Bottleneck, [3, 4, 6, 3], weight_init=args.weight_init,
                             use_bottleneck=args.bottleneck, num_classes=args.class_num, weight=args.weight)
    else:
        ## model A -> Resnet50 == pretrained
        ## model B -> Resnet50 == not pretrained
        net = network.ResNetFc(resnet_name=args.net, pretrained=args.pretrained, weight_init=args.weight_init,
                               use_bottleneck=args.bottleneck, new_cls=True, class_num=args.class_num)
    net = net.cuda()

    ## gpu
    gpus = args.gpu_id.split(',')
    if len(gpus) > 0:
        print('gpus: ', [int(i) for i in gpus])
        net = nn.DataParallel(net, device_ids=[int(i) for i in gpus])

    resume_ckpt = torch.load(args.resume_path)
    net.load_state_dict(resume_ckpt['net'])

    # torch.save(net, '../res/net_C.pkl')
    # return


    net.eval()

    file_path = '../data/predict.txt'
    pre_file = open(file_path, 'w')

    with torch.no_grad():
        for img, _, path in valid_loader:
            ## cuda
            img = img.cuda()

            feature, output = net(img)
            predict = torch.argmax(output, dim=1)

            ## write ans
            for p, label in zip(path, predict.detach().cpu().numpy()):
                s = p.split('/')[-1] + ' ' + str(label) + '\n'
                pre_file.write(s)
    pre_file.close()


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
    parser.add_argument('--vis', type=int, default=0, help="do visualization")
    parser.add_argument('--resume_path', type=str, default='')
    parser.add_argument('--debug_str', type=str, default='')

    args = parser.parse_args()

    ## prepare the path for models and visualizations
    output_str = "pretrained-%s,bottleneck-%s,augmentation-%s,weight_init-%s,opt_type-%s,momentum-%s,lr-%s,batch-%s,weight-%s,debug-%s" % (args.pretrained, args.bottleneck, args.augmentation, args.weight_init, args.opt_type, args.momentum, args.lr, args.batch_size, args.weight, args.debug_str)

    args.vis_path = "vis/" + output_str
    args.ckpt_path = "models/" + output_str

    args.test_list = osp.join(args.data_dir, 'test.txt')

    if not osp.exists(args.test_list):
        make_test_list(filename=args.test_list, path=osp.join(args.data_dir, 'test'))

    test(args)
