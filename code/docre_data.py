import numpy as np
import json
import os
import torch
from torch.utils.data import Dataset
from transformers import BertModel
import time



def load_data(prefix, data_path='./prepro_data'):
    print("Reading training data...")

    print('train', prefix)
    # data_word是每个（3053）个page，每篇中的word通过word2id映射到token的列表，长度为max_length
    data_word = np.load(os.path.join(data_path, prefix+'_word.npy'))
    # data_pos是每个实体在vertexSet中对应的实体的序号，长度为这篇文档中max_length
    data_pos = np.load(os.path.join(data_path, prefix+'_pos.npy'))
    # data_ner是代表每个word的实体类型通过ner2id映射的列表，长度为512/1024
    data_ner = np.load(os.path.join(data_path, prefix+'_ner.npy'))
    # 每个单词的字母通过char2id映射到数字
    data_char = np.load(os.path.join(data_path, prefix+'_char.npy'))
    # 有几个字段 na_triple代表没有关系的实体三元组，labels代表头尾实体和关系，vertexSet和json里相同，title是page的标题，Ls代表每个句子在page中的起始位置
    file = json.load(open(os.path.join(data_path, prefix+'.json')))

    print("Finish reading")

    return data_word, data_pos, data_ner, data_char, file
        

class DocREDataset(Dataset):
    def __init__(self, prefix : str, data_path : str):
        t1 = time.time()
        print('start')
        self.data_word, self.data_pos, self.data_ner, self.data_char, self.file = load_data(prefix=prefix, data_path=data_path)

        l = len(self.data_word)
        assert(len(self.data_word) == len(self.data_pos) == len(self.data_ner) == len(self.file))

        self.data = []

        maxNumEntity = 0
        
        for i in range(l):
            words = self.data_word[i] #这篇文档中单词映射到token的列表
            lenWords = self.file[i]['Ls'][-1]
            labels = self.file[i]['labels'] #这篇文档中的标签
            naTriple = self.file[i]['na_triple']
            numEntity = len(self.file[i]['vertexSet']) #这篇文档中实体的数量
            if numEntity > maxNumEntity:
                maxNumEntity = numEntity
            segmentID = []
            for j in range(1, len(self.file[i]['Ls'])):
                segmentID = segmentID + [0 if (j - 1) % 2 == 0 else 1] * (self.file[i]['Ls'][j] - self.file[i]['Ls'][j - 1])
            segmentID = segmentID + (1024 - len(segmentID)) * [0]
            segmentID = np.array(segmentID).astype(np.int32)

            pageData = []
            triples = []

            for label in labels:
                triple = [label['h'], label['t'], label['r']]
                triples.append(triple)
                pass

            for x in range(numEntity):
                for y in range(numEntity):
                    if x != y:
                        if [x, y] not in naTriple:
                            for triple in triples:
                                if triple[0] == x and triple[1] == y:
                                    relation = triple[2]
                        else:
                            relation = None

                        entityPos = []
                        headEntity = self.file[i]['vertexSet'][x]
                        tailEntity = self.file[i]['vertexSet'][y]
                        for headEntityMention in headEntity:
                            # sent_id = headEntityMention['sent_id']
                            start = headEntityMention['pos'][0]
                            end = headEntityMention['pos'][1]
                            entityPos.append([start, end, 'h'])
                        for tailEntityMention in tailEntity:
                            # sent_id = tailEntityMention['sent_id']
                            start = tailEntityMention['pos'][0]
                            end = tailEntityMention['pos'][1]
                            entityPos.append([start, end, 't'])
                        
                        replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos = self.replaceMention(words=words, entityPos=entityPos, segmentID=segmentID, lenWords=lenWords)
                        replacedWords = torch.from_numpy(replacedWords)
                        mask = torch.from_numpy(mask)
                        replacedSegmentID = torch.from_numpy(replacedSegmentID)
                        headMentionPos = torch.from_numpy(headMentionPos)
                        tailMentionPos = torch.from_numpy(tailMentionPos)
                        # 返回：插入了标记的文档，插入了标记后的段落ID，头实体提及的标记位置，尾实体提及的标记位置，头实体ID，尾实体ID，关系编号
                        pageData.append([replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos, x, y, relation])
            
            # pageData = pageData + (maxPairLength - len(pageData) - 1) * [[]] + [[len(pageData)]]          
            self.data.append(pageData)
            pass
        t2 = time.time()
        print(maxNumEntity)
        print(f'finished after {t2-t1}')
        pass

    def replaceMention(self, words, entityPos, segmentID, lenWords):
        sortedPos = sorted(entityPos, key=lambda x : x[0])
        replacedSegmentID = np.array([]).astype(np.int32)
        replacedWords = np.array([]).astype(np.int32)
        headMentionPos = np.array([]).astype(np.int32)
        tailMentionPos = np.array([]).astype(np.int32)
        lastEnd = 0
        for start, end, s in sortedPos:  #start、end都是在原文档中实体提及的开始和结束位置
            insertPos = 0
            if start < lastEnd:  # 表明出现了重叠实体提及
                insertPos = len(replacedWords) - (lastEnd - start)     #要在哪里插入重叠提及的标记
                replacedWords = np.concatenate((replacedWords[:insertPos], np.array([1008]).astype(np.int32), replacedWords[insertPos:]))
                replacedSegmentID = np.concatenate((replacedSegmentID[:insertPos], np.array([segmentID[start]]).astype(np.int32), replacedSegmentID[insertPos:]))
                # lastEnd = end
            else:
                replacedWords = np.concatenate((replacedWords, words[lastEnd:start]))
                insertPos = len(replacedWords)
                replacedWords = np.concatenate((replacedWords, np.array([1008]).astype(np.int32), words[start:end]))
                replacedSegmentID = np.concatenate((replacedSegmentID, segmentID[lastEnd:start], np.array([segmentID[start]]).astype(np.int32), segmentID[start:end]))
                lastEnd = end
            lenWords = lenWords + 1  # 每插入一个标记，文档长度就会+1
            if s == 'h':
                headMentionPos = np.append(headMentionPos, insertPos)
            else:
                tailMentionPos = np.append(tailMentionPos, insertPos)
        
        replacedWords = np.concatenate((replacedWords, words[lastEnd:])).astype(np.int32)
        replacedSegmentID = np.concatenate((replacedSegmentID, segmentID[lastEnd:])).astype(np.int32)
        
        if len(replacedWords) > 1024:
            replacedWords = replacedWords[0:1024]
            replacedSegmentID = replacedSegmentID[0:1024]
        else:
            replacedWords = np.pad(replacedWords, (0, 1024 - len(replacedWords)), mode='constant', constant_values=0)
            replacedSegmentID = np.pad(replacedSegmentID,  (0, 1024 - len(replacedSegmentID)), mode='constant', constant_values=0)

        ones = np.ones(lenWords, dtype=np.int32)
        zeros = np.zeros(1024 - lenWords, dtype=np.int32)
        mask = np.concatenate((ones, zeros))

        return replacedWords, mask, replacedSegmentID, headMentionPos, tailMentionPos
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
        

if __name__ == '__main__':
    d = DocREDataset('dev_train', './prepro_data', '../pretrained/')
    x = d.__getitem__(114)
    y = d.__getitem__(514)
    z = d.__getitem__(1919)
    a = d.__getitem__(810)
    pass