import torch
import torch.nn.functional as F
import math
import argparse
import numpy as np
import time
import os
from tqdm import tqdm
from model import DSAN
import data_loader
import torchvision
from model import *
from torch.utils.tensorboard import SummaryWriter

import utils
def load_data(root_path, src, tar, batch_size):
    kwargs = {'num_workers': 1, 'pin_memory': True}
    loader_src = data_loader.load_training(root_path, src, batch_size, kwargs)
    loader_tar = data_loader.load_training(root_path, tar, batch_size, kwargs)
    loader_tar_test = data_loader.load_testing(root_path, tar, batch_size * 4, kwargs)
    return loader_src, loader_tar, loader_tar_test

def train(args):
    dataloaders = load_data(args.root_path, args.src, args.tar, args.batch_size)
    args.num_classes = len(dataloaders[0].dataset.classes)
    num_classes = args.num_classes
    if args.backbone == 'resnet101':
        feature_extractor = ResBase101().cuda()
    else:
        feature_extractor = ResBase50().cuda()
    if args.bottleneck:
        bottle = nn.Linear(2048, args.bottleneck_dim).cuda()
        cls_fc = nn.Linear(args.bottleneck_dim, num_classes).cuda()
    else:
        cls_fc = nn.Linear(2048, num_classes).cuda()
    args.transfer_name = f'{args.src[0].upper()}-{args.tar[0].upper()}'
    args.log_path = os.path.join(args.output_dir,args.date+".log")
    utils.set_log_path(args.log_path)
    utils.write_to_log("{}-{}".format(args.src,args.tar))
    args.log_dir = os.path.join(args.output_dir,"log")
    writer = SummaryWriter(log_dir=args.log_dir)
    utils.set_writer(writer)
    lmmd = losses.stMMD_loss(class_num=num_classes)
    if args.bottleneck:
        optimizer = torch.optim.SGD([
            {'params': feature_extractor.parameters()},
            {'params': bottle.parameters(), 'lr': args.lr[1]},
            {'params': cls_fc.parameters(), 'lr': args.lr[2]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
        model = torch.nn.Sequential(feature_extractor,bottle,cls_fc)
    else:
        optimizer = torch.optim.SGD([
            {'params': feature_extractor.parameters()},
            {'params': cls_fc.parameters(), 'lr': args.lr[1]},
        ], lr=args.lr[0], momentum=args.momentum, weight_decay=args.decay)
        model = torch.nn.Sequential(feature_extractor,cls_fc)

    if args.iter_based:
        args.max_iter = args.iter_per_epoch * args.nepoch
        args.dy_iter = args.iter_per_epoch * args.dy_epoch
        mem_fea_s = torch.rand(0, args.bottleneck_dim if args.bottleneck else 2048).cuda()
        mem_lbl_s = -torch.ones(0,dtype=torch.long).cuda()
        mem_fea_t = torch.rand(0, args.bottleneck_dim if args.bottleneck else 2048).cuda()
        mem_lbl_t = -torch.ones(0,dtype=torch.long).cuda()
        if args.sk_ratio > 0. and args.tk_ratio > 0.:
            mem_fea_ss = int(len(dataloaders[0].dataset)*args.sk_ratio)
            mem_fea_tt = int(len(dataloaders[1].dataset)*args.tk_ratio)
        else:
            mem_fea_ss = min(args.sk,len(dataloaders[0].dataset))
            mem_fea_tt = min(args.tk,len(dataloaders[1].dataset))
    else:
        args.max_iter = args.nepoch * max(len(dataloaders[0]),len(dataloaders[1]))
        args.dy_iter = args.dy_epoch * max(len(dataloaders[0]),len(dataloaders[1]))
        mem_fea = torch.rand(len(dataloaders[0].dataset) + len(dataloaders[1].dataset), args.bottleneck_dim if args.bottleneck else 2048).cuda()
        mem_lbl = -torch.ones(len(dataloaders[0].dataset) + len(dataloaders[1].dataset),dtype=torch.long).cuda() 
        source_data_num = len(dataloaders[0].dataset)

    iter_num = 0
    max_acc = 0
    stop = 0
    for epoch in range(1, args.nepoch + 1):
        stop += 1
        for index, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = args.lr[index] / math.pow((1 + 10 * (epoch - 1) / args.nepoch), 0.75)
        source_loader, target_train_loader, _ = dataloaders
        iter_source,iter_target = iter(source_loader), iter(target_train_loader)
        if args.iter_based:
            num_iter = args.iter_per_epoch
        else:
            num_iter = max(len(source_loader),len(target_train_loader))
        bar = tqdm(range(1, num_iter+1),ncols=166,ascii=True,bar_format = "{desc}|{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]{postfix}")
        for i in bar:
            iter_num += 1
            model.train()
            optimizer.zero_grad()

            data_source,label_source,index_source = iter_source.next()
            data_target,_,index_target = iter_target.next()
            if i % len(target_train_loader) == 0:
                iter_target = iter(target_train_loader)
            if i % len(source_loader) == 0:
                iter_source = iter(source_loader)
            data_source,label_source, data_target,data_target2 = data_source[0].cuda(),label_source.cuda(), data_target[0].cuda(), data_target[1].cuda()
            feature_source = feature_extractor(data_source)
            if args.bottleneck:
                feature_source = bottle(feature_source)
            pred_source = cls_fc(feature_source)
            if args.iter_based:
                mem_fea_s = torch.cat([mem_fea_s,feature_source],dim=0)
                mem_lbl_s = torch.cat([mem_lbl_s,label_source],dim=0)
                if mem_lbl_s.shape[0]>mem_fea_ss:
                    mem_fea_s = mem_fea_s[-mem_fea_ss:]
                    mem_lbl_s = mem_lbl_s[-mem_fea_ss:]
                mem_fea = torch.cat([mem_fea_s,mem_fea_t],dim=0).detach()
                mem_lbl = torch.cat([mem_lbl_s,mem_lbl_t],dim=0).detach()
            else:
                mem_fea[index_source] = feature_source
                mem_lbl[index_source] = label_source
                mem_fea = mem_fea.detach()
                mem_lbl = mem_lbl.detach()

            feature_target = feature_extractor(data_target)
            if args.bottleneck:
                feature_target = bottle(feature_target)
            pred_target = cls_fc(feature_target)
            logits_target = torch.nn.functional.softmax(pred_target,dim=1)


            #从线性改成了非线性，从0.9~0.2到0.9~0.0
            # logits_target_thres = (args.high_thres - args.low_thres) * iter_num / args.dy_iter + args.low_thres if iter_num <= args.dy_iter else 0.
            logits_target_thres = utils.get_confidence_thres(args.high_thres,args.low_thres,iter_num,args.dy_iter)
            logits_target = torch.nn.functional.softmax(pred_target,dim=1)
            logits_target_probs,logits_target_labels = logits_target.max(-1)
            high_conf_mask = logits_target_probs>=logits_target_thres
            feature_target_conf = feature_target[high_conf_mask]
            logits_target_labels_conf = logits_target_labels[high_conf_mask]
            loss_supcon = lmmd.get_contrast_loss(torch.concat([feature_source,feature_target_conf],dim=0),torch.concat([label_source,logits_target_labels_conf],dim=0))
            # logits_target_probs,logits_target_labels = logits_target.max(-1)
            # high_conf_mask = logits_target_probs>=logits_target_thres
            # feature_target_conf = feature_target[high_conf_mask]
            # logits_target_conf = logits_target[high_conf_mask]
            # loss_lmmd = lmmd.get_loss(mem_fea,feature_target_conf,mem_lbl,logits_target_conf)
            loss_lmmd = lmmd.get_loss_w(mem_fea,feature_target,mem_lbl,logits_target)
            # loss_lmmd = lmmd.get_loss(mem_fea,feature_target,mem_lbl,logits_target)
            loss_cls = losses.label_smoothing(pred_source,label_source)
            # loss_cls = F.nll_loss(F.log_softmax(pred_source,dim=-1),label_source)
            
            # target_entropy_loss= -torch.mean((logits_target * torch.log(logits_target + 1e-18)).sum(dim=1))
            # lmmd_lambd = 2 / (1 + math.exp(-10 * (iter_num) / args.max_iter)) - 1
            lmmd_lambd = utils.get_lmmd_lambd(iter_num,args.max_iter)

            loss = loss_cls + args.weight * lmmd_lambd * loss_lmmd  + 0.5 * lmmd_lambd * loss_supcon #+ lmmd_lambd * target_entropy_loss
            # loss = loss_cls + lmmd_lambd * loss_supcon #+ lmmd_lambd * target_entropy_loss
            loss_dict = {"Loss":loss,"cls":loss_cls,"lmmd":loss_lmmd,"contrast":loss_supcon}
            info_dict = {k:v.item() for k,v in loss_dict.items()}
            info_dict.update({"thres":logits_target_thres})
            str_info_dict = {k:"{:.4f}".format(v) for k,v in info_dict.items()}

            loss.backward()
            optimizer.step()
            # bar.set_description(utils.get_loss_msg(args,epoch,**loss_dict))
            # bar.set_postfix(**{k:"{:.4f}".format(v.item()) for k,v in loss_dict.items()})
            utils.write_to_tensorboard(iter_num,f'train/{args.transfer_name}',**loss_dict)
            bar.set_description("{} Epoch:[{}/{}]".format(args.transfer_name,epoch,args.nepoch))
            bar.set_postfix(**str_info_dict)

            model.eval()
            with torch.no_grad():
                feature_target = feature_extractor(data_target)
                if args.bottleneck:
                    feature_target = bottle(feature_target)
                pred_target = cls_fc(feature_target)
                logits_target = torch.nn.functional.softmax(pred_target,dim=1)
                logits_target_probs,logits_target_labels = logits_target.max(-1)
                high_conf_mask = logits_target_probs>=logits_target_thres
                if args.iter_based:
                    feature_target = feature_target[high_conf_mask]
                    logits_target_labels = logits_target_labels[high_conf_mask]
                    mem_fea_t = torch.cat([mem_fea_t,feature_target],dim=0)
                    mem_lbl_t = torch.cat([mem_lbl_t,logits_target_labels],dim=0)
                    if mem_lbl_t.shape[0]>mem_fea_tt:
                        mem_fea_t = mem_fea_t[-mem_fea_tt:]
                        mem_lbl_t = mem_lbl_t[-mem_fea_tt:]
                else:
                    if high_conf_mask.sum()>0:
                        mem_fea[index_target[high_conf_mask]+source_data_num] = feature_target[high_conf_mask]
                        mem_lbl[index_target[high_conf_mask]+source_data_num] = logits_target_labels[high_conf_mask]

        # torch.save(model,os.path.join(args.output_dir,"checkpoints",args.transfer_name+"_best.pth"))
        if args.dataset == "visda17":
            test_acc, predict, all_output, all_label, test_acc_all = utils.cal_acc_visda(model,dataloaders[-1])
            test_info = {"test acc":test_acc,"test acc all":test_acc_all}
        else:
            t_correct,test_acc,test_loss = test(model, dataloaders[-1])
            test_info = {"loss":test_loss,"acc":test_acc}
        
        if max_acc < test_acc:
            max_acc,stop = test_acc,0
            torch.save(model,os.path.join(args.output_dir,"checkpoints",args.transfer_name+"_best.pth"))
        test_info.update({"max acc":max_acc})
        # utils.write_to_tensorboard(iter_num,f'test/{args.transfer_name}',**test_info)
        test_msg = f'[{epoch}/{args.nepoch}]{args.src}-{args.tar}: current acccuracy: {100.0*test_acc:.2f}%, max accuracy: {100.0 * max_acc:.2f}%'
        if args.dataset == "visda17":
            test_msg += "\n{}".format(test_acc_all)
            msg="{} test_acc:{:.2f}% max_acc:{:2f}%".format(args.transfer_name,test_acc*100,max_acc*100)
            utils.send_to_wechat(msg=msg,title=msg)
        utils.write_to_log(test_msg,is_print=True)

        # test_summary(model,target_train_loader,epoch)

        if stop >= args.early_stop:
            early_stop_msg = f'{args.transfer_name,} Final test acc: {100. * max_acc:.2f}%'
            early_stop_brief = f'{args.transfer_name} acc: {100. * max_acc  :.2f}%'
            utils.write_to_log(early_stop_msg,is_print=True)
            utils.send_to_wechat(msg=early_stop_msg,title=early_stop_brief)
            utils.write_final_log(args.transfer_name,max_acc)
            writer.close()
            break



def get_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, help='Root path for dataset',
                        default='/data2/ywzhao/data/office31')
    parser.add_argument('--src', type=str,
                        help='Source domain', default='amazon')
    parser.add_argument('--tar', type=str,
                        help='Target domain', default='webcam')
    parser.add_argument('--num_classes', type=int,
                        help='Number of classes', default=31)
    parser.add_argument('--batch_size', type=float,
                        help='batch size', default=32)
    parser.add_argument('--nepoch', type=int,
                        help='Total epoch num', default=200)
    parser.add_argument('--lr', type=list, help='Learning rate', default=[0.001, 0.01, 0.01,0.01])
    parser.add_argument('--early_stop', type=int,
                        help='Early stoping number', default=30)
    parser.add_argument('--seed', type=int,
                        help='Seed', default=2021)
    parser.add_argument('--weight', type=float,
                        help='Weight for adaptation loss', default=1.)
    parser.add_argument('--momentum', type=float, help='Momentum', default=0.9)
    parser.add_argument('--decay', type=float,
                        help='L2 weight decay', default=5e-4)
    parser.add_argument('--bottleneck', type=str2bool,
                        nargs='?', const=True, default=True)
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('--log_interval', type=int,
                        help='Log interval', default=10)
    parser.add_argument('--gpu', type=str,
                        help='GPU ID', default='0')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda_u', default=100, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    parser.add_argument('--output_dir',default="output",type=str)
    parser.add_argument('--log_dir',default="output",type=str)
    # parser.add_argument('--tensorboard_dir',default="output",type=str)
    parser.add_argument('--K', default=5, type=float)
    parser.add_argument('--mem_momentum', default=0.1, type=float)
    parser.add_argument('--tar_par', type=float, default=0.2)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=0.7)

    parser.add_argument('--backbone',type=str,default='resnet50')
    parser.add_argument('--sk',type=int,default=-1)
    parser.add_argument('--tk',type=int,default=-1)
    parser.add_argument('--sk_ratio',type=float,default=-1.0)
    parser.add_argument('--tk_ratio',type=float,default=-1.0)
    parser.add_argument('--iter_based',action='store_true')
    parser.add_argument('--iter_per_epoch',type=int,default=500)
    parser.add_argument('--dataset',type=str,default="office31")
    parser.add_argument('--easy_margin',type=bool,default=False)
    parser.add_argument('--dy_epoch', type=int, default=20)
    parser.add_argument('--high_thres', type=float, default=0.9)
    parser.add_argument('--low_thres', type=float, default=0.7)
    parser.add_argument('--thres_g', type=float, default=-10.)
    parser.add_argument('--date', type=str, default=time.strftime('%Y-%m-%d',time.localtime(time.time())))
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    print(vars(args))
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    train(args)