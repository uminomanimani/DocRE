import numpy as np
import json
import os


def load_data(data_path, prefix):
    print("Reading training data...")

    print('train', prefix)
    # data_word是每个（3053）个page，每篇中的word通过word2id映射到token的列表，长度为512
    data_word = np.load(os.path.join(data_path, prefix+'_word.npy'))
    # data_pos是每个实体在vertexSet中对应的实体的序号，长度为512
    data_pos = np.load(os.path.join(data_path, prefix+'_pos.npy'))
    # data_ner是代表每个word的实体类型通过ner2id映射的列表，长度为512
    data_ner = np.load(os.path.join(data_path, prefix+'_ner.npy'))
    # 每个单词的字母通过char2id映射到数字
    data_char = np.load(os.path.join(data_path, prefix+'_char.npy'))
    # 有几个字段 na_triple代表没有关系的实体三元组，labels代表头尾实体和关系，vertexSet和json里相同，title是page的标题，Ls代表每个句子在page中的起始位置
    file = json.load(open(os.path.join(data_path, prefix+'.json')))

    print("Finish reading")

    return data_word, data_pos, data_ner, data_char, file
