import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer import predict
import torch.distributed as dist


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    try:
        print("\nExtract Features:")
        
        # 确保所有进程同步
        if config.distributed:
            dist.barrier()
        
        with torch.amp.autocast(device_type='cuda',enabled=config.mixed_precision):
            reference_features, reference_labels = predict(config, model, reference_dataloader, img_type="reference") 
            query_features, query_labels = predict(config, model, query_dataloader, img_type="query")
            # 同步特征提取
            if config.distributed:
                dist.barrier()
            
            torch.cuda.empty_cache()
            
            if config.distributed:
                # 收集所有进程的特征
                gathered_ref_features = [torch.zeros_like(reference_features) for _ in range(config.world_size)]
                gathered_ref_labels = [torch.zeros_like(reference_labels) for _ in range(config.world_size)]
                
                dist.all_gather(gathered_ref_features, reference_features)
                dist.all_gather(gathered_ref_labels, reference_labels)
                
                reference_features = torch.cat(gathered_ref_features, dim=0)
                reference_labels = torch.cat(gathered_ref_labels, dim=0)
            
                # 收集所有进程的特征
                gathered_query_features = [torch.zeros_like(query_features) for _ in range(config.world_size)]
                gathered_query_labels = [torch.zeros_like(query_labels) for _ in range(config.world_size)]
                
                dist.all_gather(gathered_query_features, query_features)
                dist.all_gather(gathered_query_labels, query_labels)
                
                query_features = torch.cat(gathered_query_features, dim=0)
                query_labels = torch.cat(gathered_query_labels, dim=0)
        
        # 只在主进程中计算评估指标
        if not config.distributed or config.rank == 0:
            print("Compute Scores:")
            # 将特征移到CPU计算相似度，减少GPU内存压力
            reference_features = reference_features.cpu()
            query_features = query_features.cpu()
            reference_labels = reference_labels.cpu()
            query_labels = query_labels.cpu()
            
            r1 = calculate_scores(query_features, 
                                reference_features, 
                                query_labels, 
                                reference_labels, 
                                step_size=step_size, 
                                ranks=ranks)
            
        else:
            r1 = 0.0
            
        # 同步评估结果
        if config.distributed:
            r1 = torch.tensor(r1).to(config.device)
            dist.broadcast(r1, 0)
            r1 = r1.item()
            
        # cleanup and free memory
        if cleanup:
            del reference_features, reference_labels, query_features, query_labels
            if config.distributed:
                del gathered_ref_features, gathered_ref_labels
                del gathered_query_features, gathered_query_labels
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            
        return r1
    
    except Exception as e:
        print(f"Error in evaluate: {e}")
        if cleanup:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        raise


def calc_sim(config,
                        model,
                        reference_dataloader,
                        query_dataloader, 
                        ranks=[1, 5, 10],
                        step_size=1000,
                        cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,img_type="reference") 
    query_features, query_labels = predict(config, model, query_dataloader,img_type="query")
    
    print("Compute Scores Train:")
    r1 =  calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict



def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    hit_rate = 0.0
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i][0]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        # mask for semi pos
        mask = torch.ones(R)
        for near_pos in query_labels_np[i][1:]:
            mask[ref2index[near_pos]] = 0
        
        # calculate hit rate
        hit = (higher_sim * mask).sum()
        if hit < 1:
            hit_rate += 1.0
                
    
    results = results/ Q * 100.
    hit_rate = hit_rate / Q * 100
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))
    string.append('Hit_Rate: {:.4f}'.format(hit_rate))             
        
    print(' - '.join(string)) 

    return results

def calculate_scores_train(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    query_labels_np = query_labels[:,0].cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
        
    results = results/ Q * 100.

    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))           
        
    print(' - '.join(string)) 

    return results[0]
   

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):

    query_labels = query_labels[:,0]
    
    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)


    # there might be more ground views for same sat view
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+2, dim=1)


    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict
