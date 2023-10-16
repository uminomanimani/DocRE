import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel
from docre_data import DocREDataset

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

def cf(batch):
    return batch

def collate_fn(batch):
    label = []
    new_batch = []
    for items in batch:  #len(items)是实体对的数量. label map的数量和batch_size的大小相同
        new_item = []
        labelMap = []
        for item in items:
            new_item.clear()
            new_item.append(item[0])
            new_item.append(item[1])
            new_item.append(item[2])
            new_item.append(item[3])
            new_item.append(item[4])
        new_batch.append(new_item)

        # new_item = []
        # replacedWords = [item[0] for item in items]
        # mask = [item[1] for item in items]
        # replacedSegmentID = [item[2] for item in items]
        # headMentionPos = [item[3] for item in items]
        # tailMentionPos = [item[4] for item in items]

        # new_batch.append([replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos])
        # headEntityID = [item[5] for item in items]
        # tailEntityID = [item[6] for item in items]
        # relation = [item[7] for item in items]

        # labelMap = torch.Tensor(size=(len(relation) + 1, len(relation) + 1))

    return new_batch, label

def encodeV2(bert, batch):

    pass     


class MyModel(nn.Module):
    def __init__(self, bertPath : str, replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos, headEntityID, tailEntityID) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bertPath)

        for param in self.bert.parameters():
            param.requires_grad = False
        

if __name__ == "__main__":
    # model = MyModel()
    dataset = DocREDataset('dev_train', './prepro_data')
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn)
    for i, batch in enumerate(dataloader):
        pass