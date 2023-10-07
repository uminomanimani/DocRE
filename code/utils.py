import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import BertModel


def load_data(prefix, data_path='./prepro_data'):
    print("Reading training data...")

    print('train', prefix)
    # data_word是每个（3053）个page，每篇中的word通过word2id映射到token的列表，长度为max_length
    data_word = np.load(os.path.join(data_path, prefix+'_word.npy'))
    # data_pos是每个实体在vertexSet中对应的实体的序号，长度为这篇文档中max_length
    data_pos = np.load(os.path.join(data_path, prefix+'_pos.npy'))
    # data_ner是代表每个word的实体类型通过ner2id映射的列表，长度为512
    data_ner = np.load(os.path.join(data_path, prefix+'_ner.npy'))
    # 每个单词的字母通过char2id映射到数字
    data_char = np.load(os.path.join(data_path, prefix+'_char.npy'))
    # 有几个字段 na_triple代表没有关系的实体三元组，labels代表头尾实体和关系，vertexSet和json里相同，title是page的标题，Ls代表每个句子在page中的起始位置
    file = json.load(open(os.path.join(data_path, prefix+'.json')))

    print("Finish reading")

    return data_word, data_pos, data_ner, data_char, file

def encode(pathBert : str, replacedWords : list, headMentionPos : list, tailMentionPos : list):
    bert = BertModel.from_pretrained(pathBert)
    for param in bert.parameters():
        param.requires_grad = False
    _replacedWords = torch.Tensor(replacedWords)
        

class DocREDataset(Dataset):
    def __init__(self, prefix : str, data_path : str):
        self.data_word, self.data_pos, self.data_ner, self.data_char, self.file = load_data(prefix=prefix, data_path=data_path)
        l = len(self.data_word)
        assert(len(self.data_word) == len(self.data_pos) == len(self.data_ner) == len(self.file))
        
        for i in range(l):
            words = self.data_word[i] #这篇文档中单词映射到token的列表
            labels = self.file[i]['labels'] #这篇文档中的标签
            numEntity = len(self.file[i]['vertexSet']) #这篇文档中实体的数量
            segmentID = []
            for j in range(1, len(self.file[i]['Ls'])):
                segmentID = segmentID + [j - 1] * self.file[i]['Ls'][j]
            segmentID = segmentID + (1024 - len(segmentID)) * [0]

            for x in numEntity:
                for y in numEntity:
                    if x != y:
                        entityPos = []
                        headEntity = self.file[i]['vertexSet'][x]
                        tailEntity = self.file[i]['vertexSet'][y]
                        for headEntityMention in headEntity:
                            sent_id = headEntityMention['sent_id']
                            start = headEntityMention['pos'][0]
                            end = headEntityMention['pos'][1]
                            entityPos.append([start + self.file[i]['Ls'][sent_id], end + self.file[i]['Ls'][sent_id], 'h'])
                        for tailEntityMention in tailEntity:
                            sent_id = tailEntityMention['sent_id']
                            start = tailEntityMention['pos'][0]
                            end = tailEntityMention['pos'][1]
                            entityPos.append([start + self.file[i]['Ls'][sent_id], end + self.file[i]['Ls'][sent_id], 't'])
                        
                        replacedWords, replacedSegmentID, headMentionPos, tailMentionPos = self.replaceMention(words=words, entityPos=entityPos, segmentID=segmentID)

        pass

    def replaceMention(self, words, entityPos, segmentID):
        sorted_pos = sorted(entityPos, key=lambda x : x[0])
        replacedSegmentID = []
        replacedWords = []
        headMentionPos = []
        tailMentionPos = []
        lastEnd = 0
        for start, end, s in sorted_pos:
            replacedWords += (words[lastEnd:start] + [-1 if s == 'h' else -2])
            replacedSegmentID += ((segmentID[lastEnd:start]) + segmentID[lastEnd])
            lastEnd = end
        
        replacedWords = replacedWords + words[lastEnd:]

        for i in range(len(replacedWords)):
            if replacedWords[i] == -1:
                replacedWords[i] = 20
                headMentionPos.append(i)
            if replacedWords[i] == -2:
                replacedWords[i] = 20
                tailMentionPos.append(i)
        
        return replacedWords, replacedSegmentID, headMentionPos, tailMentionPos
        

if __name__ == '__main__':
    d = DocREDataset('dev_train', './prepro_data')