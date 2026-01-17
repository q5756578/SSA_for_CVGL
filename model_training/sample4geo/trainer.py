import time
import torch
from tqdm import tqdm
from .utils import AverageMeter
import torch.amp as amp  # 使用新的amp API
import torch.nn.functional as F
from sample4geo.loss import MultiClassDiceLoss

def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    model.train()
    losses = AverageMeter()
    losses_bev = AverageMeter()
    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    optimizer.zero_grad(set_to_none=True)
    
    loss_bev_ce = torch.nn.CrossEntropyLoss()
    loss_bev_dice = MultiClassDiceLoss(num_classes=8)
    
    for query, reference, reference_bev, ids in bar:
        #print('1111',query.shape)
        #print('2222',reference.shape)
        #print('33333',reference_bev.shape)
        #print(reference_bev.shape)
        try:
            query = query.to(config.device, non_blocking=True)
            reference = reference.to(config.device, non_blocking=True)
            reference_bev = reference_bev.to(config.device, non_blocking=True)
            #query_depth = query_depth.to(config.device, non_blocking=True)
            
            reference_bev = reference_bev.squeeze(1)
            
            if scaler:
                with amp.autocast('cuda'):
                    features1, features2, bev_layout_feat = model(query, reference, None, True) 
                    #print('bev_layout_feat',bev_layout_feat.shape)
                    logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()
                    logit_scale = logit_scale * 1.5 #// 提升对于样本区分程度
                    loss = loss_function(features1, features2, logit_scale)
                    loss_bev = 0.5 * loss_bev_ce(bev_layout_feat, reference_bev) + \
                             0.5 * loss_bev_dice(F.softmax(bev_layout_feat,dim=1), reference_bev)
                    
                    #loss_bev = loss_bev_dice(bev_layout_feat, reference_bev) 
                    
                #scaler.scale(loss ).backward()
                scaler.scale( loss + loss_bev).backward()
                if config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                features1, features2,bev_layout_feat = model(query, reference, None, True)
                logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()
                loss = loss_function(features1, features2, logit_scale)
                loss_bev = 0.5 * loss_bev_ce(bev_layout_feat, reference_bev) + \
                             0.5 * loss_bev_dice(bev_layout_feat, reference_bev)
                
                (loss +  loss_bev).backward()
                if config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            
            losses.update(loss.item())
            losses_bev.update(loss_bev.item())
            
            if config.verbose :
                monitor = {
                    "loss": "{:.4f}".format(loss.item()),
                    "loss_bev": "{:.4f}".format(loss_bev.item()),
                    "loss_avg": "{:.4f}".format(losses.avg),
                    "loss_bev_avg": "{:.4f}".format(losses_bev.avg),
                    "lr": "{:.6f}".format(optimizer.param_groups[0]['lr'])
                }
                bar.set_postfix(ordered_dict=monitor)
                
        except Exception as e:
            print(f"Error in training iteration: {e}")
            continue
    
    if config.verbose:
        bar.close()
    
    return losses.avg




'''
def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    model.train()
    losses = AverageMeter()
    losses_bev = AverageMeter()
    
    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
    
    optimizer.zero_grad(set_to_none=True)
    
    loss_bev_ce = torch.nn.CrossEntropyLoss()
    loss_bev_dice = MultiClassDiceLoss(num_classes=2)
    
    for query, reference, reference_bev, ids in bar:
        try:
            query = query.to(config.device, non_blocking=True)
            reference = reference.to(config.device, non_blocking=True)
            reference_bev = reference_bev.to(config.device, non_blocking=True)
            #query_depth = query_depth.to(config.device, non_blocking=True)
            
            reference_bev = reference_bev.squeeze(1)
            
            if scaler:
                with amp.autocast('cuda'):
                    #features1, features2, bev_layout_feat = model(query, reference, None, True)
                    features1, features2 = model(query, reference, None, False)
                    logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()
                    #loss = loss_function(features1, features2, logit_scale)
                    #loss_bev = 0.5 * loss_bev_ce(bev_layout_feat, reference_bev) + \
                            #  0.5 * loss_bev_dice(bev_layout_feat, reference_bev)
                
                scaler.scale(loss ).backward()
                if config.clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                features1, features2 = model(query, reference, None, False)
                logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()
                loss = loss_function(features1, features2, logit_scale)
                #loss_bev = 0.5 * loss_bev_ce(bev_layout_feat, reference_bev) + \
                #          0.5 * loss_bev_dice(bev_layout_feat, reference_bev)
                
                (loss + 0.01 ).backward()
                if config.clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            
            losses.update(loss.item())
            #losses_bev.update(loss_bev.item())
            
            if config.verbose:
                bar.set_postfix(ordered_dict={
                    "loss": f"{loss.item():.4f}",
                    #"loss_bev": f"{loss_bev.item():.4f}",
                    "loss_avg": f"{losses.avg:.4f}",
                    #"loss_bev_avg": f"{losses_bev.avg:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })
                
        except Exception as e:
            print(f"Error in training iteration: {e}")
            continue
    
    if config.verbose:
        bar.close()
    
    return losses.avg

def train(config, model, dataloader, loss_function, optimizer, scheduler=None, scaler=None):
    model.train()
    losses = AverageMeter()
    
    if config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader

    optimizer.zero_grad(set_to_none=True)
    
    accumulation_steps = getattr(config, "accumulation_steps", 1)  # 默认为 1（无梯度累加）
    accumulation_counter = 0  # 记录当前累积步数

    for query, reference, reference_bev, query_depth,ids in bar:
        try:
            query = query.to(config.device, non_blocking=True)
            reference = reference.to(config.device, non_blocking=True)
            reference_bev = reference_bev.to(config.device, non_blocking=True)
            #query_depth = query_depth.to(config.device, non_blocking=True)

            reference_bev = reference_bev.squeeze(1)

            if scaler:
                with amp.autocast('cuda'):
                    features1, features2 = model(query, reference, None, False)
                    logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()
                    loss = loss_function(features1, features2, logit_scale) / accumulation_steps  # 归一化 Loss
                
                scaler.scale(loss).backward()
            else:
                features1, features2 = model(query, reference, None, False)
                logit_scale = model.module.logit_scale.exp() if isinstance(model, torch.nn.DataParallel) else model.logit_scale.exp()
                loss = loss_function(features1, features2, logit_scale) / accumulation_steps  # 归一化 Loss

                loss.backward()

            accumulation_counter += 1  # 更新累积计数器

            if accumulation_counter % accumulation_steps == 0:  # 每 accumulation_steps 进行一次更新
                if config.clip_grad:
                    if scaler:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)  # 清空梯度
                if scheduler is not None:
                    scheduler.step()

            losses.update(loss.item() * accumulation_steps)  # 还原原始 Loss 方便观察
            
            if config.verbose:
                bar.set_postfix(ordered_dict={
                    "loss": f"{loss.item() * accumulation_steps:.4f}",
                    "loss_avg": f"{losses.avg:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
                })

        except Exception as e:
            print(f"Error in training iteration: {e}")
            continue

    if config.verbose:
        bar.close()

    return losses.avg
'''


def predict(train_config, model, dataloader,img_type=None):
    
    model.eval()
    
    # wait before starting progress bar
    time.sleep(0.1)
    
    if train_config.verbose:
        bar = tqdm(dataloader, total=len(dataloader))
    else:
        bar = dataloader
        
    img_features_list = []
    
    ids_list = []
    with torch.no_grad():
        
        for img, ids in bar:
        
            ids_list.append(ids)
            
            with amp.autocast('cuda'):
         
                img = img.to(train_config.device)
                img_feature = model(img,img,img_type,False)
            
                # normalize is calculated in fp32
                if train_config.normalize_features:
                    img_feature = F.normalize(img_feature, dim=-1)
            
            # save features in fp32 for sim calculation
            img_features_list.append(img_feature.to(torch.float32))
      
        # keep Features on GPU
        img_features = torch.cat(img_features_list, dim=0) 
        ids_list = torch.cat(ids_list, dim=0).to(train_config.device)
        
    if train_config.verbose:
        bar.close()
        
    return img_features, ids_list




