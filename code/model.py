import numpy as np
import torch
import torch.nn as nn
from transformers import BertModel

def encode(bert, replacedWords, replacedSegmentID, headMentionPos, tailMentionPos):
    lenPage = np.argwhere(replacedWords == 0)[0][0] if len(np.argwhere(replacedWords == 0)) > 0 else 1024
    mask = np.array([1] * lenPage + (1024 - lenPage) * [0]).astype(np.int32)
    # replacedWords = torch.from_numpy(replacedWords).unsqueeze(0)
    # replacedSegmentID = torch.from_numpy(replacedSegmentID).unsqueeze(0)
    mask = torch.from_numpy(mask).unsqueeze(0)
    # print(replacedWords.shape)
    # print(replacedSegmentID.shape)
    # print(mask.shape)

    hiddenStates = bert(input_ids=replacedWords, attention_mask=mask, token_type_ids=replacedSegmentID)[0]
    hiddenStates = torch.squeeze(input=hiddenStates, dim=0)
    headMentionPos = torch.from_numpy(headMentionPos)
    tailMentionPos = torch.from_numpy(tailMentionPos)
    headMentionRepresentition = hiddenStates[headMentionPos]
    tailMentionRepresentation = hiddenStates[tailMentionPos]
    headEntityRepresentation = torch.logsumexp(headMentionRepresentition, dim=0)
    tailEntityRepresentation = torch.logsumexp(tailMentionRepresentation, dim=0)
    return headEntityRepresentation, tailEntityRepresentation

class MyModel(nn.Module):
    def __init__(self, bertPath : str) -> None:
        super().__init__()
        self.bert = BertModel.from_pretrained(bertPath)

        for param in self.bert.parameters():
            param.requires_grad = False
        


    


if __name__ == "__main__":
    x = MyModel()