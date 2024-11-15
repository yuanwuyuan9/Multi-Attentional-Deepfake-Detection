import time
import numpy
import torch
from torch.utils.data import DataLoader
from datasets.dataset import mm_data
import cv2
from config import train_config

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# GPU settings
assert torch.cuda.is_available()
from models.MAT import MAT
from AGDA import AGDA
import cv2
from utils import dist_average,ACC
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
# GPU settings
assert torch.cuda.is_available()

#torch.autograd.set_detect_anomaly(True)
def main_worker(config,MAX_acc):
    numpy.random.seed(1234567)
    torch.manual_seed(1234567)
    torch.cuda.manual_seed(1234567)
    train_dataset = mm_data(root=r"D:\ff_dataset\train_dataset_frames", phase='train', **config.train_dataset)
    validate_dataset = mm_data(root=r"D:\ff_dataset\val_dataset_frames", phase='val', **config.val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=config.batch_size,shuffle=True)
    start_epoch = 0
    net = MAT(**config.net_config)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate, betas=config.adam_betas,weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step,gamma=config.scheduler_gamma)
    model_dict=torch.load("model_dict/model_best")
    sche_dict = torch.load("model_dict/sche_dict")
    optim_dict=torch.load("model_dict/optimizer_dict")
    net.load_state_dict(model_dict,strict=False)
    optimizer.load_state_dict(optim_dict)
    scheduler.load_state_dict(sche_dict)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()
    numpy.random.seed(1234567)
    torch.manual_seed(1234567)
    torch.cuda.manual_seed(1234567)
    start_epoch = 0

    for i in config.freeze:
        if 'backbone' in i:
            net.net.requires_grad_(False)
        elif 'attention' in i:
            net.attentions.requires_grad_(False)
        elif 'feature_center' in i:
            net.auxiliary_loss.alpha=0
        elif 'texture_enhance' in i:
            net.texture_enhance.requires_grad_(False)
        elif 'fcs' in i:
            net.projection_local.requires_grad_(False)
            net.project_final.requires_grad_(False)
            net.ensemble_classifier_fc.requires_grad_(False)
        else:
            if 'xception' in str(type(net.net)):
                for j in net.net.seq:
                    if j[0]==i:
                        for t in j[1]:
                            t.requires_grad_(False)
            
            if 'EfficientNet' in str(type(net.net)):
                if i=='b0':
                    net.net._conv_stem.requires_grad_(False)
                stage_map=net.net.stage_map
                for c in range(len(stage_map)-2,-1,-1):
                    if not stage_map[c]:
                        stage_map[c]=stage_map[c+1]
                for c1,c2 in zip(stage_map,net.net._blocks):
                    if c1==i:
                        c2.requires_grad_(False)
    net=net.to('cuda')
    AG=AGDA(**config.AGDA_config).to('cuda')
    torch.cuda.empty_cache()
    loss=0
    accuracy_train=0
    accuracy_val=0
    for epoch in range(start_epoch, config.epochs):
        print(f"-------------------第{epoch+1}轮训练开始--------------------------")
        train_time1=time.time()
        loss,accuracy_train=run(data_loader=train_loader,net=net,optimizer=optimizer,config=config,AG=AG,phase='train')
        train_time2= time.time()
        print("cost_train_time:{}".format(train_time1-train_time2))
        print(f"-------------------第{epoch + 1}轮验证开始--------------------------")
        val_time1 = time.time()
        ll,accuracy_val=run(data_loader=validate_loader,net=net,optimizer=optimizer,config=config,phase='valid')
        val_time2 = time.time()
        print("cost_val_time:{}".format(train_time1 - train_time2))
        file=open("D:\model_oral\log.txt","a+")
        file.write(f"This is {epoch+1} round:"+'\n')
        file.write(f"train_time:{train_time2-train_time1},val_time:{val_time2-val_time1}"+'\n')
        file.write(f"last_train_loss:{loss}------train_acc:{accuracy_train}-----val_acc:{accuracy_val}"+'\n')
        if(accuracy_val>MAX_acc):
            MAX_acc=accuracy_val
            torch.save(net.state_dict(),"model_dict/model_best")
            print(f"---------目前最好模型准确率:{accuracy_val},在第{epoch+1}轮产生-------")
        file.write(f"Now‘s bestmodel_ACC:{MAX_acc}"+'\n')
        net.auxiliary_loss.alpha*=config.alpha_decay
        scheduler.step()
        torch.save(net.state_dict(),"model_dict/model_dict")
        torch.save(scheduler.state_dict(),"model_dict/sche_dict")
        torch.save(optimizer.state_dict(),"model_dict/optimizer_dict")
        print(f"已经保存第{epoch+1}轮模型")
def train_loss(loss_pack,config):
    if 'loss' in loss_pack:
        return loss_pack['loss']
    loss=config.ensemble_loss_weight*loss_pack['ensemble_loss']+config.aux_loss_weight*loss_pack['aux_loss']
    if config.AGDA_loss_weight!=0:
        loss+=config.AGDA_loss_weight*loss_pack['AGDA_ensemble_loss']+config.match_loss_weight*loss_pack['match_loss']
    return loss
    
def run(data_loader,net,optimizer,config,AG=None,phase='train'):
    if config.AGDA_loss_weight==0:
        AG=None
    start_time = time.time()
    if phase=='train':
        net.train()
    else: net.eval()
    ACC_total=0
    loss_total = 0
    nums=0
    a=time.time()
    for i, (X, y) in enumerate(data_loader):
        X = X.to('cuda',non_blocking=True)
        y = y.to('cuda',non_blocking=True)
        with torch.set_grad_enabled(phase=='train'):
            loss_pack=net(X,y,train_batch=True,AG=AG)
        if phase=='train':
            batch_loss = train_loss(loss_pack,config)
            batch_loss.backward()
            loss_total+=batch_loss.item()
            optimizer.step()
            optimizer.zero_grad()
        with torch.no_grad():
            if config.feature_layer=='logits':
                loss_pack['acc']=ACC(loss_pack['logits'],y)
            else:
                loss_pack['ensemble_acc'],num=ACC(loss_pack['ensemble_logit'],y)
                ACC_total+=loss_pack['ensemble_acc']

                nums+=num
        if((i%100==0)&(i!=0)):
            acc_total = ACC_total/(i + 1)
            loss_now=loss_total/(i+1)
            print(f"----------目前的损失{loss_now}-----------")
            print(f"-----------目前的准确率{acc_total}-----------")
            print(f"-----目前测试个数{i*4}---------测试准确个数:{nums}--------------")
            c=time.time()
            print(f"time:{c-a}")
            a=time.time()
    ACC_total=ACC_total/(i+1)
    loss_total=loss_total/(i+1)
    return loss_total,ACC_total
    # end of this epoch
feature_layer='b2'
name='EFB4_ALL_c23_trunc_b2'
Config=train_config(name,['ff-all-c23','efficientnet-b4'],attention_layer='b5',feature_layer=feature_layer,epochs=20,batch_size=4,augment='augment1')
Max_acc=0.9628328010559082
main_worker(Config,Max_acc)
