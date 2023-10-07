import numpy as np
import json
import os
from torch.utils.data import Dataset


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

def encode(page : list):

    pass

class DocREDataset(Dataset):
    def __init__(self, prefix : str, data_path : str):
        self.data_word, self.data_pos, self.data_ner, self.data_char, self.file = load_data(prefix=prefix, data_path=data_path)
        l = len(self.data_word)
        assert(len(self.data_word) == len(self.data_pos) == len(self.data_ner) == len(self.file))
        
        for i in range(l):
            words = self.data_word[i] #这篇文档中单词映射到token的列表
            labels = self.file[i]['labels'] #这篇文档中的标签
            num_entity = len(self.file[i]['vertexSet']) #这篇文档中实体的数量
            for x in num_entity:
                for y in num_entity:
                    if x != y:
                        entity_pos = []
                        head_entity = self.file[i]['vertexSet'][x]
                        tail_entity = self.file[i]['vertexSet'][y]
                        for head_entity_mention in head_entity:
                            sent_id = head_entity_mention['sent_id']
                            start = head_entity_mention['pos'][0]
                            end = head_entity_mention['pos'][1]
                            entity_pos.append([start + self.file[i]['Ls'][sent_id], end + self.file[i]['Ls'][sent_id], 'h'])
                        for tail_entity_mention in tail_entity:
                            sent_id = tail_entity_mention['sent_id']
                            start = tail_entity_mention['pos'][0]
                            end = tail_entity_mention['pos'][1]
                            entity_pos.append([start + self.file[i]['Ls'][sent_id], end + self.file[i]['Ls'][sent_id], 't'])
        pass

    def replaceMention(self, words, entity_pos):
        sorted_pos = sorted(entity_pos, key=lambda x : x[0])
        
        pass

if __name__ == '__main__':
    d = DocREDataset('dev_train', './prepro_data')