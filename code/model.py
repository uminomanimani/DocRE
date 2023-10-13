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
def encodeV2(bert, replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos, headEntityID, tailEntityID, relation):
    # item的每个元素的第一个维度是两两配对的实体对的数量
    for item in zip(replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos, headEntityID, tailEntityID):
        # featureMap = torch.Tensor(size=(len(item) + 1, len(item) + 1, 768))
        hiddenStates = bert(input_id=item[0], attention_mask=item[1], token_type_ids=item[2])[0]

    pass

import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    max_length = max(len(sample) for sample in batch)

    words_for_pad = [0] * 1024
    mask_for_pad = [0] * 1024
    segmentID_for_pad = [0] * 1024
    

    for sample in batch:
        for words, mask, segmentID, headMentionPos, tailMentionPos, headID, tailID, relation in sample:

            pass

        


class MyModel(nn.Module):
    def __init__(self, bertPath : str, replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos, headEntityID, tailEntityID) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bertPath)

        for param in self.bert.parameters():
            param.requires_grad = False
        

if __name__ == "__main__":
    # model = MyModel()
    dataset = DocREDataset('dev_train', './prepro_data', '../pretrained/')
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn)
    for batch in dataloader:
        b = batch
        pass