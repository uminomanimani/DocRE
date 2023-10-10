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

def encode(bert, replacedWords, replacedSegmentID, headMentionPos, tailMentionPos):
    #bert = BertModel.from_pretrained(pathBert)
    #for param in bert.parameters():
    #    param.requires_grad = False
    lenPage = np.argwhere(replacedWords == 0)[0][0] if len(np.argwhere(replacedWords == 0)) > 0 else 512
    mask = np.array([1] * lenPage + (512 - lenPage) * [0]).astype(np.int32)
    replacedWords = torch.from_numpy(replacedWords).unsqueeze(0)
    replacedSegmentID = torch.from_numpy(replacedSegmentID).unsqueeze(0)
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
        

class DocREDataset(Dataset):
    def __init__(self, prefix : str, data_path : str, bertPath : str):
        self.data_word, self.data_pos, self.data_ner, self.data_char, self.file = load_data(prefix=prefix, data_path=data_path)
        
        self.bert = BertModel.from_pretrained(bertPath)
        for param in self.bert.parameters():
            param.requires_grad = False

        l = len(self.data_word)
        assert(len(self.data_word) == len(self.data_pos) == len(self.data_ner) == len(self.file))

        self.data = []
        
        for i in range(l):
            if i == 282:
                pass
            words = self.data_word[i] #这篇文档中单词映射到token的列表
            # lenWords = self.file[i]['Ls'][-1]
            labels = self.file[i]['labels'] #这篇文档中的标签
            numEntity = len(self.file[i]['vertexSet']) #这篇文档中实体的数量
            segmentID = []
            for j in range(1, len(self.file[i]['Ls'])):
                segmentID = segmentID + [0 if (j - 1) % 2 == 0 else 1] * (self.file[i]['Ls'][j] - self.file[i]['Ls'][j - 1])
            segmentID = segmentID + (512 - len(segmentID)) * [0]

            # featureMap = torch.Tensor(size=(numEntity, numEntity, 2, 768))
            pageData = []

            for x in range(numEntity):
                for y in range(numEntity):
                    if x != y:
                        entityPos = []
                        headEntity = self.file[i]['vertexSet'][x]
                        tailEntity = self.file[i]['vertexSet'][y]
                        for headEntityMention in headEntity:
                            sent_id = headEntityMention['sent_id']
                            start = headEntityMention['pos'][0]
                            if start == 116:
                                pass
                            end = headEntityMention['pos'][1]
                            entityPos.append([start, end, 'h'])
                        for tailEntityMention in tailEntity:
                            sent_id = tailEntityMention['sent_id']
                            start = tailEntityMention['pos'][0]
                            end = tailEntityMention['pos'][1]
                            entityPos.append([start, end, 't'])
                        
                        replacedWords, replacedSegmentID, headMentionPos, tailMentionPos = self.replaceMention(words=words, entityPos=entityPos, segmentID=segmentID)
                        pageData.append((replacedWords, replacedSegmentID, headMentionPos, tailMentionPos))
                        # headEntityRepresentation, tailEntityRepresentation = encode(bert=self.bert, replacedWords=replacedWords, replacedSegmentID=replacedSegmentID, 
                        #                         headMentionPos=headMentionPos, tailMentionPos=tailMentionPos)
                        # featureMap[x][y][0] = headEntityRepresentation
                        # featureMap[x][y][1] = tailEntityRepresentation
            
            self.data.append(pageData)
            print(f'added page {i}')
        pass

    def replaceMention(self, words, entityPos, segmentID):
        sortedPos = sorted(entityPos, key=lambda x : x[0])
        replacedSegmentID = np.array([]).astype(np.int32)
        replacedWords = np.array([]).astype(np.int32)
        headMentionPos = np.array([]).astype(np.int32)
        tailMentionPos = np.array([]).astype(np.int32)
        # print(replacedWords.shape)
        # print(words.shape)
        lastEnd = 0
        for start, end, s in sortedPos:
            # replacedWords += words[lastEnd:start]
            replacedWords = np.concatenate((replacedWords, words[lastEnd:start])).astype(np.int32)
            #  + [-1 if s == 'h' else -2])
            replacedWords = np.append(replacedWords, -1 if s == 'h' else -2).astype(np.int32)
            # replacedSegmentID += (segmentID[lastEnd:start] + [segmentID[lastEnd]])
            replacedSegmentID = np.concatenate((replacedSegmentID, segmentID[lastEnd:start])).astype(np.int32)
            replacedSegmentID = np.append(replacedSegmentID, segmentID[start]).astype(np.int32)
            lastEnd = end
        
        replacedWords = np.concatenate((replacedWords, words[lastEnd:])).astype(np.int32)
        replacedSegmentID = np.concatenate((replacedSegmentID, segmentID[lastEnd:])).astype(np.int32)

        replacedWords = np.pad(replacedWords, (0, 512 - len(replacedWords)), mode='constant', constant_values=0)
        replacedSegmentID = np.pad(replacedSegmentID,  (0, 512 - len(replacedSegmentID)), mode='constant', constant_values=0)

        for i in range(len(replacedWords)):
            if replacedWords[i] == -1:
                replacedWords[i] = 20
                headMentionPos = np.append(headMentionPos, i)
            if replacedWords[i] == -2:
                replacedWords[i] = 20
                tailMentionPos = np.append(tailMentionPos, i)
        
        return replacedWords, replacedSegmentID, headMentionPos, tailMentionPos
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
        

if __name__ == '__main__':
    d = DocREDataset('dev_train', './prepro_data', '../pretrained/')
    x = d.__len__()