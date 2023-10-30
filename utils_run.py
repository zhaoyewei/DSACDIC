#%%
import os
import torch
from torch import nn
import numpy as np
from sklearn.metrics import confusion_matrix
import requests
import math
from tqdm import tqdm
import torch.nn.functional as F
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
LOG_PATH = "output"
WRITEER = None
def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
class WeightEMA(object):
    def __init__(self, model, ema_model, lr=1e-3,alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    #(32//3)==10, [10,10,10]
    groups = [batch // (nu + 1)] * (nu + 1)
    #0,1->groups:[10,11,11]
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    #[0],[10],[21],[32]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    #2
    nu = len(xy) - 1
    #32,2
    offsets = interleave_offsets(batch, nu)
    #[0],[10],[21],[32]
    # v in xy:a,b,c
    #v[0:10],v[10:21],v[21:32]
    #xy=[[a[0:10],a[10:21],a[21:32]],
    #    [b[0:10],b[10:21],b[21:32]],
    #    [c[0:10],c[10:21],c[21:32]]]
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    #1,2
    #xy=[[a[0:10],b[10:21],c[21:32]],
    #    [b[0:10],a[10:21],b[21:32]],
    #    [c[0:10],c[10:21],a[21:32]]]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def interleave_offsets_unbalance(batch, nu):
    #(32//3)==10, [10,10,10]
    groups = [batch // (nu + 1)] * (nu + 1)
    #0,1->groups:[10,11,11]
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    #[0],[10],[21],[32]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave_unbalance(xy, batches):
    #2
    nu = len(xy) - 1
    #32,2
    all_offsets = []
    for batch in batches:
        all_offsets.append(interleave_offsets_unbalance(batch, nu))
    #[0],[10],[21],[32]
    # v in xy:a,b,c
    #v[0:10],v[10:21],v[21:32]
    #xy=[[a[0:10],a[10:21],a[21:32]],
    #    [b[0:10],b[10:21],b[21:32]],
    #    [c[0:10],c[10:21],c[21:32]]]
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v,offsets in zip(xy,all_offsets)]
    #1,2
    #xy=[[a[0:10],b[10:21],c[21:32]],
    #    [b[0:10],a[10:21],b[21:32]],
    #    [c[0:10],c[10:21],a[21:32]]]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def print_args(args):
    log_str = ("==========================================\n")
    log_str += ("==========       config      =============\n")
    log_str += ("==========================================\n")
    for arg, content in args.__dict__.items():
        log_str += ("{}:{}\n".format(arg, content))
    log_str += ("\n==========================================\n")
    print(log_str)
    args.out_file.write(log_str+'\n')
    args.out_file.flush()

def cal_fea(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            inputs, labels = iter_test.next()
            inputs = inputs.cuda()
            feas, outputs = model(inputs)
            if start_test:
                all_feas = feas.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_feas = torch.cat((all_feas, feas.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_feas, all_label

def cal_acc(loader, model, flag=True, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    # outputs = model.predict(inputs)
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy, predict, all_output, all_label

def cal_acc_visda(model,loader,flag=False, fc=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm(range(len(loader)),ncols=50):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            if flag:
                _, outputs = model(inputs)
            else:
                if fc is not None:
                    feas, outputs = model(inputs)
                    outputs = fc(feas)
                else:
                    outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
    acc = matrix.diagonal()/matrix.sum(axis=1) * 100
    aacc = acc.mean() / 100
    aa = [str(np.round(i, 2)) for i in acc]
    acc = ' '.join(aa)

    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return aacc, predict, all_output, all_label, acc
    
def lr_scheduler(optimizer, init_lr, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def set_log_path(log_path,new_file=False):
    global LOG_PATH
    LOG_PATH = log_path
    if new_file and os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

def write_final_log(tp,acc):
    with open(LOG_PATH.split(".")[0]+"_final.log","a") as f:
        f.write(f'{tp}:{acc*100:.2f}%\n')

def write_to_log(msg,is_print=False):
    global LOG_PATH
    with open(LOG_PATH,"a") as f:
        f.write(msg+"\r\n")
    if is_print:
        print(msg)

def get_confidence_thres(high_thres,low_thres,iter_num,decrease_iter,g=-10):
    # return (1/(1-(high_thres-low_thres)*(decrease_iter-iter_num)/decrease_iter-low_thres)-1/(1-low_thres))*(high_thres-low_thres)/(1/(1-high_thres)-1/(1-low_thres)) + low_thres
    # thres =  high_thres - (1/(2 / (1 + math.exp(g * (iter_num) / decrease_iter))) - 0.5 )* (high_thres - low_thres) / 0.5 
    # return (1/(2 / (1 + math.exp(g * (iter_num) / decrease_iter))) - 0.5 )* (high_thres - low_thres) / 0.5 + low_thres if iter_num <= decrease_iter else low_thres
    return  (decrease_iter - iter_num) / decrease_iter * (high_thres - low_thres) + low_thres if iter_num <= decrease_iter else low_thres

def get_lmmd_lambd(iter_num,max_iter):
    return 2 / (1 + math.exp(-10 * (iter_num) / max_iter)) - 1
    # return 2 - 2 / (1 + math.exp(-10 * (iter_num) / max_iter))

def get_loss_msg(args,epoch,**losses):
    msg = f'{args.src[0].upper()}-{args.tar[0].upper()} Epoch:[{epoch:3d}/{args.nepoch:3d}] '
    msg += ",".join([f'{k}:{v.item():.4f}' for k,v in losses.items()])
    return msg
def set_writer(writer):
    global WRITEER
    WRITEER = writer
def write_to_tensorboard(iter_num,tag,**info):
    global WRITEER
    WRITEER.add_scalars(tag,info,iter_num)

def send_to_wechat(msg,title="DA结果"):
    try:
        headers = {"Content-Type":"application/json"}
        url = "https://wxpusher.zjiecode.com/api/send/message"
        data = {
        "appToken":"AT_HNhCFubNfbO9GQXgDBgueWcfJYBGJfNc",
        "content":msg+'''<table border="1px solid black">
            <tbody><tr><td>Office31</td><td style="width: 600px;height:200px"><img width="100%" height="100%" src="https://s1.ax1x.com/2023/02/27/pp9xX9S.png"></td></tr>
            <tr><td>Office-Home</td><td style="width: 600px;height:200px"><img width="100%" height="100%" src="https://s1.ax1x.com/2023/02/27/pp9zuH1.png"></td></tr>
            <tr><td>ImageCLEF-DA</td><td style="width: 600px;height:200px"><img width="100%" height="100%" src="https://s1.ax1x.com/2023/02/27/pp9zQN6.png"></td></tr>
            <tr><td>VISDA-2017</td><td style="width: 600px;height:200px"><img width="100%" height="100%" src="https://s1.ax1x.com/2023/02/27/pp9zUHI.png"></td></tr>
            <tr><td>Adaptiope</td><td style="width: 600px;height:200px"><img width="100%" height="100%" src="https://s1.ax1x.com/2023/03/24/pp0TXb6.png"></td></tr>
        </tbody></table>''',
        "summary":title,
        "contentType":2,
        "topicIds":[ 
            123
        ],
        "uids":[
            "UID_Lx1oilJos9A5vfj1wZNpHxetteVO"
        ],
        }
        response = requests.post(url,headers=headers,json=data)
    except Exception as e:
        pass

def test(model,dataloader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(dataloader,ncols=50):
            data, target = data.cuda(), target.cuda()
            pred = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(F.log_softmax(pred, dim=1), target).item()
            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(dataloader)
        test_acc = correct / len(dataloader.dataset)
        test_msg = f'Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(dataloader.dataset)} ({100. * correct / len(dataloader.dataset):.2f}%)'
        write_to_log(test_msg,is_print=True)
    return correct,test_acc,test_loss

def test_summary(model,dataloader,epoch):
    model.eval()
    # correct1,correct2 = 0.,0.
    preds1,preds2,gt = [],[],[]
    for [d0,d1],t,_ in tqdm(dataloader,ncols=100,ascii=True):
        d0,d1,t = d0.cuda(),d1.cuda(),t.cuda()
        d0t = model(d0)
        d1t = model(d1)
        preds1.append(nn.functional.softmax(d0t,dim=1).data)
        preds2.append(nn.functional.softmax(d1t,dim=1).data)
        gt.append(t.data)
    preds1 = torch.vstack(preds1)
    preds2 = torch.vstack(preds2)
    probs1,labels1 = preds1.max(-1)
    probs2,labels2 = preds2.max(-1)
    gt = torch.hstack(gt)
    write_to_log("[Epoch:{}]".format(epoch),is_print=True)
    for thres in np.arange(0.1,1,0.01):
        #高置信占全部的比例
        high_prob_mask = (probs1 >= thres) & (probs2 >= thres)
        high_prob_prop = high_prob_mask.sum()/labels1.shape[0]
        eq_mask = high_prob_mask & (labels1 == labels2)
        eq_prop = eq_mask.sum()/high_prob_mask.sum() if high_prob_mask.sum() != 0 else 0.
        right_mask = eq_mask & (labels1 == gt)
        if eq_mask.sum().item() != 0:
            accuracy = right_mask.sum().item()/eq_mask.sum().item()
        else:
            accuracy = 0.
        acc_msg = "Thres:{:.2f},Acc:{:.2f}%,high prob:{:.2f}%,eq:{:.2f}%".format(thres,accuracy*100,high_prob_prop*100,eq_prop*100)
        write_to_log(acc_msg,is_print=True)


def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='r', target_color='b',show=False):
    """
    Visualize features from different domains using t-SNE.
    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'
    """
    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([source_color, target_color]), s=5)
    plt.axis('off')
    # plt.axes().get_xaxis().set_visible(False)
    # plt.axes().get_yaxis().set_visible(False)
    plt.savefig(filename)
    if show:
        plt.show()

def plot_tsne():
    import data_loader 
    from model import ResBase50
    feature_extractor = ResBase50().cuda()
    root_path = "/data3/ywzhao/data/office-home"
    appendix = "ours"
    output_dir = "output/2023-04-17_train_target"
    appendix = "dsan"
    output_dir = "output/2023-04-17_train_target_ablation_dsan"
    checkpoint_dirs = os.path.join(output_dir,"checkpoints")
    batch_size = 128
    for checkpoint_name in tqdm(os.listdir(checkpoint_dirs)):
        # model = torch.load(os.path.join(checkpoint_dirs,checkpoint_name),map_location={"cuda":"cpu"}).eval()
        model = torch.load(os.path.join(checkpoint_dirs,checkpoint_name)).eval()
        feature_extractor = model[0].cuda()
        src = checkpoint_name.split("_")[0].split("-")[0]
        tar = checkpoint_name.split("_")[0].split("-")[1]
        src = {"A":"Art","C":"Clipart","P":"Product","R":"Real_World"}[src]
        tar = {"A":"Art","C":"Clipart","P":"Product","R":"Real_World"}[tar]
        source_loader = data_loader.load_testing(root_path, src, batch_size,{})
        target_laader = data_loader.load_testing(root_path, tar, batch_size,{})
        iter_source, iter_target = iter(source_loader),iter(target_laader)
        with torch.no_grad():
            sfeats,tfeats = torch.Tensor(),torch.Tensor()
            for _ in range(8):
                sdata,_ = iter_source.next()
                tdata,_ = iter_target.next()
                sdata,tdata = sdata.cuda(),tdata.cuda()
                sfeat,tfeat = feature_extractor(sdata),feature_extractor(tdata)
                sfeat,tfeat = sfeat.to("cpu"),tfeat.to("cpu")
                sfeats = torch.concat([sfeat,sfeats],axis=0)
                tfeats = torch.concat([tfeat,tfeats],axis=0)
            visualize(sfeats,tfeats,os.path.join(output_dir,"images",checkpoint_name.split(".")[0]+"_"+appendix+".jpg"))

#%%
if __name__ == "__main__":
    from collections import defaultdict
    import data_loader 
    from model import ResBase50
    feature_extractor = ResBase50().cuda()
    root_path = "/data3/ywzhao/data/office-home"
    batch_size = 128
    domain_dict = {"A":"Art","C":"Clipart","P":"Product","R":"Real_World"}
    total_dict = defaultdict(lambda:{})
    appendix = "ours"
    for output_dir in  [ "output/2023-08-02_train_target3","output/2023-04-17_train_target_ablation_dsan"]:
        checkpoint_dirs = os.path.join(output_dir,"checkpoints")
        for checkpoint_name in tqdm(os.listdir(checkpoint_dirs)):
            # model = torch.load(os.path.join(checkpoint_dirs,checkpoint_name),map_location={"cuda":"cpu"}).eval()
            model = torch.load(os.path.join(checkpoint_dirs,checkpoint_name)).eval()
            feature_extractor = model[0].cuda()
            _src = checkpoint_name.split("_")[0].split("-")[0]
            _tar = checkpoint_name.split("_")[0].split("-")[1]
            src = domain_dict[_src]
            tar = domain_dict[_tar]
            # source_loader = data_loader.load_testing(root_path, src, batch_size,{})
            target_laader = data_loader.load_testing(root_path, tar, batch_size,{})
            # iter_source, iter_target = iter(source_loader),iter(target_laader)
            iter_target = iter(target_laader)
            with torch.no_grad():
                sfeats,tfeats = torch.Tensor(),torch.Tensor()
                slabels,tlabels = torch.Tensor(),torch.Tensor()
                spreds,tpreds = torch.Tensor(),torch.Tensor()

                for i in range(len(target_laader)):
                    # sdata,slabel = iter_source.next()
                    tdata,tlabel = iter_target.next()
                    # sdata,tdata = sdata.cuda(),tdata.cuda()
                    tdata = tdata.cuda()
                    # sfeat,tfeat = feature_extractor(sdata),feature_extractor(tdata)
                    tfeat = feature_extractor(tdata)
                    # spred = model[2](model[1](sfeat)).argmax(-1).to("cpu")
                    tpred = model[2](model[1](tfeat)).argmax(-1).to("cpu")
                    tfeat = tfeat.to("cpu")
                    # sfeat,tfeat = sfeat.to("cpu"),tfeat.to("cpu")
                    # sfeats = torch.concat([sfeat,sfeats],axis=0)
                    tfeats = torch.concat([tfeat,tfeats],axis=0)
                    # slabels = torch.concat([slabels,slabel],axis=0)
                    # spreds = torch.concat([spreds,spred],axis=0)
                    tlabels = torch.concat([tlabels,tlabel],axis=0)
                    tpreds = torch.concat([tpreds,tpred],axis=0)
            class_sim = []
            for label in torch.unique(tlabels):
                cur_feats = tfeats[(tlabels==label)&(tpreds==label)]
                if cur_feats.shape[0]>0:
                    cur_class_sim = []
                    for i in range(cur_feats.shape[0]):
                        _mask = torch.ones((cur_feats.shape[0],),dtype=torch.bool)
                        _mask[i] = False
                        # cur_similarity = F.cosine_similarity(cur_feats,cur_feats[i:i+1])[_mask]
                        cur_similarity = (((cur_feats-cur_feats[i:i+1]))**2)[_mask]
                        cur_class_sim.append(cur_similarity)
                    cur_class_sim = torch.vstack(cur_class_sim)
                    # cur_class_mask = torch.ones(cur_class_sim.shape,dtype=torch.bool)
                    # cur_class_mask[torch.eye(cur_class_sim.shape[0],dtype=torch.long)] = False
                    # cur_class_sim = cur_class_sim[cur_class_mask]
                    # print(cur_similarity.shape,cur_class_sim.shape)
                    cur_class_mean = cur_class_sim.mean()
                    class_sim.append(cur_class_mean)
            print(output_dir.split("_")[-1])
            print(f"{_src}-{_tar}:",sum(class_sim)/len(class_sim))

            # total_dict[f"{_src}-{_tar}"]["sfeat"] = sfeats
            # total_dict[f"{_src}-{_tar}"]["tfeat"] = tfeats
            # total_dict[f"{_src}-{_tar}"]["slabel"] = slabels
            # total_dict[f"{_src}-{_tar}"]["tlabel"] = tlabels
            


    