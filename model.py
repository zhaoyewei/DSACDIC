import torch
import torch.nn as nn
import losses
import math
import argparse
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
from utils import init_weights
import torchvision
from utils import *
import torch.utils.model_zoo as model_zoo
from torch import nn
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

class DSAN(nn.Module):
    def __init__(self, num_classes=31, bottle_neck=True,max_iter=1000):
        super(DSAN, self).__init__()
        self.feature_layers = ResBase50()
        self.lmmd_loss = losses.ori_LMMD_loss(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            bottleneck_dim = 256
            self.bottle = nn.Linear(2048, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            bottleneck_dim = 2048
            self.cls_fc = nn.Linear(2048, num_classes)
        self.num_class = num_classes
        self.max_iter = max_iter
        self.iter_num = 0
        self.ad_net = AdversarialNetwork(bottleneck_dim*num_classes, 1024, max_iter=self.max_iter).cuda()
    # def init_cate_dict(self,source_loader,target_loader,num_classes,cate_len):
    #     with torch.no_grad():
    #         source_cate_dict = {idx:[] for idx in range(num_classes)}
    #         iter_data = iter(source_loader)
    #         dict_counter = 0
    #         dict_index_set = set()
    #         while True:
    #             data_source, label_source = next(iter_data)
    #             data_source,label_source = data_source.to(self.device),label_source.to(self.device)
    #             source,source_stages = self.base_network(data_source)
    #             if self.use_bottleneck:
    #                 source = self.bottleneck_layer(source)
    #             source_clf = self.classifier_layer(source)
    #             #B,D
    #             source = source[source_clf.argmax(-1)==label_source]
    #             label_source = label_source[source_clf.argmax(-1)==label_source]
    #             for source_f,class_idx in zip(source[:],label_source.int().tolist()):
    #                 if len(source_cate_dict[class_idx])<cate_len:
    #                     source_cate_dict[class_idx].append(F.normalize(source_f.detach(),dim=-1))
    #                 if len(source_cate_dict[class_idx])>=cate_len and class_idx not in dict_index_set:
    #                     dict_index_set.add(class_idx)
    #                     dict_counter += 1
    #             if dict_counter >= num_classes:
    #                 break
    #         source_cate_dict = torch.stack([torch.stack(v) for k,v in source_cate_dict.items()])

    #         target_cate_dict = {idx:[] for idx in range(num_classes)}
    #         iter_data = iter(target_loader)
    #         dict_counter = 0
    #         dict_index_set = set()
    #         thres = 0.9
    #         while True:
    #             data_target, _ = next(iter_data)
    #             data_target = data_target.to(self.device)
    #             target,target_stages = self.base_network(data_target)
    #             if self.use_bottleneck:
    #                 target = self.bottleneck_layer(target)
    #             target_clf = self.classifier_layer(target)
    #             #B,D
    #             label_target = target_clf.argmax(-1)
    #             target = target[label_target>thres]
    #             label_target = label_target[label_target>thres]
    #             for target_f,class_idx in zip(target[:],label_target.int().tolist()):
    #                 if len(target_cate_dict[class_idx])<cate_len:
    #                     target_cate_dict[class_idx].append(F.normalize(target_f.detach(),dim=-1))
    #                 if len(target_cate_dict[class_idx])>=cate_len and class_idx not in dict_index_set:
    #                     dict_index_set.add(class_idx)
    #                     dict_counter += 1
    #             if dict_counter >= num_classes:
    #                 break
    #             thres -= 0.01
    #         target_cate_dict = torch.stack([torch.stack(v) for k,v in target_cate_dict.items()])
    #         cate_dict = torch.concat([source_cate_dict,target_cate_dict],dim=1)
    #         idx = torch.randperm(cate_dict.shape[1])
    #         cate_dict = cate_dict[:,idx]
    #         return cate_dict.to(self.device)

    # def update_cate_dict(self,source_clf,source_label,source_feature,target_clf,target_feature):
    #     with torch.no_grad():
    #         new_cate_dict = torch.zeros_like(self.cate_dict)
    #         source_feature = source_feature[source_clf.argmax(-1)==source_label]
    #         source_label = source_label[source_clf.argmax(-1)==source_label]
    #         class_idx_set = set()
    #         for source_f,class_idx in zip(source_feature[:],source_label.int().tolist()):
    #             if class_idx not in class_idx_set:
    #                 class_idx_set.add(class_idx)
    #                 new_cate_dict[:,1:] = self.cate_dict[:,:-1].detach()
    #                 new_cate_dict[:,0] = F.normalize(source_f.detach(),dim=-1)
    #         target_label = target_clf[target_clf.argmax(-1)>0.8].argmax(-1)
    #         target_feature = target_feature[target_clf.argmax(-1)>0.8]
    #         for target_f,class_idx in zip(target_feature[:],target_label.int().tolist()):
    #             if class_idx not in class_idx_set:
    #                 class_idx_set.add(class_idx)
    #                 new_cate_dict[:,1:] = self.cate_dict[:,:-1].detach()
    #                 new_cate_dict[:,0] = F.normalize(target_f.detach(),dim=-1)

    #         return new_cate_dict.to(self.device)

    # def get_logits_thres(self,max_thres=0.8,min_thres=0.2):
    #     p = self.iter_num / self.max_iter
    #     return max_thres - (max_thres - min_thres) * p

    # def step(self):
    #     self.iter_num+=1

    def forward(self, source, target, s_label):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)

        target = self.feature_layers(target)
        if self.bottle_neck:
            target = self.bottle(target)
        t_pred = self.cls_fc(target)
        t_logits = torch.nn.functional.softmax(t_pred,dim=1)

        # entropy = None
        # features = torch.cat((source, target), dim=0)
        # outputs = torch.cat((s_pred,t_pred), dim=0)
        # softmax_out = nn.Softmax(dim=1)(outputs)
        # eff = calc_coeff(self.iter_num, max_iter=self.max_iter)
        # cdan_loss = losses.CDAN([features, softmax_out], self.ad_net, entropy, eff)

        # logits_thres = self.get_logits_thres()
        # t_logits_max = t_logits.max(-1)
        # t_logits_filter_bool = t_logits_max.values>logits_thres
        # t_logits_filter_index = torch.arange(0,t_logits.size()[0])[t_logits_filter_bool]
        # t_logits = t_logits[t_logits_filter_index]
        # target = target[t_logits_filter_index]
        loss_lmmd = self.lmmd_loss.get_loss(source, target, s_label,t_logits)
        # ad_net = utils.AdversarialNetwork(args.bottleneck_dim*class_num, 1024, max_iter=args.max_iter).cuda() 
        # entropy = loss.Entropy(softmax_out)
        # self.step()
        return s_pred, loss_lmmd

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)

class VGG16Base(nn.Module):
  def __init__(self):
    super(VGG16Base, self).__init__()
    model_vgg = torchvision.models.vgg16(pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)
    self.in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

class ResBase18(nn.Module):
    def __init__(self):
        super(ResBase18, self).__init__()
        model_resnet = torchvision.models.resnet18(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase34(nn.Module):
    def __init__(self):
        super(ResBase34, self).__init__()
        model_resnet = torchvision.models.resnet34(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase50(nn.Module):
    def __init__(self):
        super(ResBase50, self).__init__()
        model_resnet50 = torchvision.models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResBase101(nn.Module):
    def __init__(self):
        super(ResBase101, self).__init__()
        model_resnet101 = torchvision.models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class ResClassifier(nn.Module):
    def __init__(self, class_num, feature_dim,bottleneck=True,bottleneck_dim=256):
        super(ResClassifier, self).__init__()
        self.bottleneck = bottleneck
        if bottleneck:
            self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
        else:
            bottleneck_dim = feature_dim
            self.fc = nn.Linear(bottleneck_dim, class_num)
        # self.bottleneck.apply(init_weights)
        # self.fc.apply(init_weights)

    def forward(self, x):
        if self.bottleneck:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x,y

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size, max_iter=10000):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = max_iter

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet101(pretrained=False,**kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model