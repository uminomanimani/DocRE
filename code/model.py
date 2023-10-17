from typing import Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel
from docre_data import DocREDataset
import math

# def encode(bert, replacedWords, replacedSegmentID, headMentionPos, tailMentionPos):
#     lenPage = np.argwhere(replacedWords == 0)[0][0] if len(np.argwhere(replacedWords == 0)) > 0 else 1024
#     mask = np.array([1] * lenPage + (1024 - lenPage) * [0]).astype(np.int32)
#     # replacedWords = torch.from_numpy(replacedWords).unsqueeze(0)
#     # replacedSegmentID = torch.from_numpy(replacedSegmentID).unsqueeze(0)
#     mask = torch.from_numpy(mask).unsqueeze(0)
#     # print(replacedWords.shape)
#     # print(replacedSegmentID.shape)
#     # print(mask.shape)

#     hiddenStates = bert(input_ids=replacedWords, attention_mask=mask, token_type_ids=replacedSegmentID)[0]
#     hiddenStates = torch.squeeze(input=hiddenStates, dim=0)
#     headMentionPos = torch.from_numpy(headMentionPos)
#     tailMentionPos = torch.from_numpy(tailMentionPos)
#     headMentionRepresentition = hiddenStates[headMentionPos]
#     tailMentionRepresentation = hiddenStates[tailMentionPos]
#     headEntityRepresentation = torch.logsumexp(headMentionRepresentition, dim=0)
#     tailEntityRepresentation = torch.logsumexp(tailMentionRepresentation, dim=0)
#     return headEntityRepresentation, tailEntityRepresentation

# 这些参数的的第一个维度都是batch_size
# 数据的格式(batch_size = 16)：
# [
#     [[replacedPage0, mask, replacedSegmentID, headMentionPos, tailMentionPos, headEntityID, tailEntityID, relation], [...], [...], ...]
#     [[replacedPage1...], [...], ...]
#     [...]
#     ...
#     [[replacedPage15...], [...], ...]
#]

def collate_fn(batch):
    label = []
    for items in batch:  #len(items)是实体对的数量. label map的数量和batch_size的大小相同
        numEntity = math.ceil(math.sqrt(len(items)))
        labelMap = torch.zeros(size=(numEntity, numEntity), dtype=torch.int32)
        for item in items:
            if item[7] != None:
                labelMap[item[5]][item[6]] = item[7]
        label.append(labelMap)
    return batch, label

class Encode:
    def __init__(self, bert):
        self.bert = bert
        pass 

    def __call__(self, batch):
        featureMaps = []
        for items in batch:
            numEntity = math.ceil(math.sqrt(len(items)))
            replacedWords = []
            masks = []
            segmentIDs = []
            headMentionPos = []
            tailMentionPos = []
            headEntityID = []
            tailEntityID = []
            for item in items:
                replacedWords.append(item[0])
                masks.append(item[1])
                segmentIDs.append(item[2])
                headMentionPos.append(item[3])
                tailMentionPos.append(item[4])
                headEntityID.append(item[5])
                tailEntityID.append(item[6])
            # replacedWords = torch.tensor(replacedWords, dtype=torch.int32)
            # masks = torch.tensor(masks, dtype=torch.int32)
            # segmentIDs = torch.tensor(segmentIDs, dtype=torch.int32)
            hiddenStates = self.bert(input_ids=replacedWords, attention_mask=masks, token_type_ids=segmentIDs)[0]
            featureMap = torch.zeros(size=(numEntity, numEntity, 768 * 2))
            for headMention, tailMention, headID, tailID, hiddenState in zip(headMentionPos, 
                                                                             tailMentionPos, 
                                                                             headEntityID, 
                                                                             tailEntityID, 
                                                                             hiddenStates):
                headMentionRepresentation = hiddenState[headMention]
                tailMentionRepresentation = hiddenState[tailMention]
                headEntityRepresentation = torch.logsumexp(headMentionRepresentation, dim=0)
                tailEntityRepresentation = torch.logsumexp(tailMentionRepresentation, dim=0)
                
        pass  


class DocREModel(nn.Module):
    def __init__(self, bertPath : str) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bertPath)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.encode = Encode(bert=self.bert)

    
    def forward(self, batch):
        featureMaps = self.encode(batch=batch)
        return None
        pass

        

if __name__ == "__main__":
    model = DocREModel('../pretrained')
    dataset = DocREDataset('dev_test', './prepro_data')
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    for data, label in dataloader:
        x = model(data)
        pass