import numpy as np
import torch
import math



def test_all_users(model, batch_size, item_num, test_data_pos, user_pos, top_k):
    
    predictedIndices = []
    GroundTruth = []
    for u in test_data_pos:
        batch_num = item_num // batch_size
        batch_user = torch.Tensor([u]*batch_size).long().cuda()
        st, ed = 0, batch_size
        for i in range(batch_num):
            batch_item = torch.Tensor([i for i in range(st, ed)]).long().cuda()
            pred = model(batch_user, batch_item)
            if i == 0:
                predictions = pred
            else:
                predictions = torch.cat([predictions, pred], 0)
            st, ed = st+batch_size, ed+batch_size
        ed = ed - batch_size
        batch_item = torch.Tensor([i for i in range(ed, item_num)]).long().cuda()
        batch_user = torch.Tensor([u]*(item_num-ed)).long().cuda()
        pred = model(batch_user, batch_item)
        predictions = torch.cat([predictions, pred], 0)
        test_data_mask = [0] * item_num
        if u in user_pos:
            for i in user_pos[u]:
                test_data_mask[i] = -9999
        predictions = predictions + torch.Tensor(test_data_mask).float().cuda()
        _, indices = torch.topk(predictions, top_k[-1])
        indices = indices.cpu().numpy().tolist()
        predictedIndices.append(indices)
        GroundTruth.append(test_data_pos[u])
    precision, recall, NDCG, MRR = compute_acc(GroundTruth, predictedIndices, top_k)
    return precision, recall, NDCG, MRR
    
def compute_acc(GroundTruth, predictedIndices, topN):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
        
        precision.append(sumForPrecision / len(predictedIndices))
        recall.append(sumForRecall / len(predictedIndices))
        NDCG.append(sumForNdcg / len(predictedIndices))
        MRR.append(sumForMRR / len(predictedIndices))
        
    return precision, recall, NDCG, MRR

