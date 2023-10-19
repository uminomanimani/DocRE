import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from docre_data import DocREDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn.functional as F

# 每条包含以下几个部分：
# input_ids:单词转为token的列表，长度和文章长度相同
# entity_pos:实体的提及的开始和结束的位置
# labels: 实体对之间的关系标签的独热编码，数量为n * (n - 1)，每个元素长度为关系的数量（97）
# hts : 两两组合的实体对的集合，长度为n * (n - 1)，n为实体的数量
def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    entity_pos = [f["entity_pos"] for f in batch]

    labels = [f["labels"] for f in batch]
    hts = [f["hts"] for f in batch]
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output

class Encode:
    def __init__(self, bert, config):
        self.bert = bert
        self.config = config
        pass 

    def process_long_input(self, model, input_ids, attention_mask):
        # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]
        n, c = input_ids.size()
        start_tokens = torch.tensor(start_tokens).to(input_ids)
        end_tokens = torch.tensor(end_tokens).to(input_ids)
        len_start = start_tokens.size(0)
        len_end = end_tokens.size(0)
        if c <= 512:
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            sequence_output = output[0]
            attention = output[-1][-1]
        else:
            new_input_ids, new_attention_mask, num_seg = [], [], []
            seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
            for i, l_i in enumerate(seq_len):
                if l_i <= 512:
                    new_input_ids.append(input_ids[i, :512])
                    new_attention_mask.append(attention_mask[i, :512])
                    num_seg.append(1)
                else:
                    input_ids1 = torch.cat([input_ids[i, :512 - len_end], end_tokens], dim=-1)
                    input_ids2 = torch.cat([start_tokens, input_ids[i, (l_i - 512 + len_start): l_i]], dim=-1)
                    attention_mask1 = attention_mask[i, :512]
                    attention_mask2 = attention_mask[i, (l_i - 512): l_i]
                    new_input_ids.extend([input_ids1, input_ids2])
                    new_attention_mask.extend([attention_mask1, attention_mask2])
                    num_seg.append(2)
            input_ids = torch.stack(new_input_ids, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
            sequence_output = output[0]
            attention = output[-1][-1]
            i = 0
            new_output, new_attention = [], []
            for (n_s, l_i) in zip(num_seg, seq_len):
                if n_s == 1:
                    output = F.pad(sequence_output[i], (0, 0, 0, c - 512))
                    att = F.pad(attention[i], (0, c - 512, 0, c - 512))
                    new_output.append(output)
                    new_attention.append(att)
                elif n_s == 2:
                    output1 = sequence_output[i][:512 - len_end]
                    mask1 = attention_mask[i][:512 - len_end]
                    att1 = attention[i][:, :512 - len_end, :512 - len_end]
                    output1 = F.pad(output1, (0, 0, 0, c - 512 + len_end))
                    mask1 = F.pad(mask1, (0, c - 512 + len_end))
                    att1 = F.pad(att1, (0, c - 512 + len_end, 0, c - 512 + len_end))

                    output2 = sequence_output[i + 1][len_start:]
                    mask2 = attention_mask[i + 1][len_start:]
                    att2 = attention[i + 1][:, len_start:, len_start:]
                    output2 = F.pad(output2, (0, 0, l_i - 512 + len_start, c - l_i))
                    mask2 = F.pad(mask2, (l_i - 512 + len_start, c - l_i))
                    att2 = F.pad(att2, [l_i - 512 + len_start, c - l_i, l_i - 512 + len_start, c - l_i])
                    mask = mask1 + mask2 + 1e-10
                    output = (output1 + output2) / mask.unsqueeze(-1)
                    att = (att1 + att2)
                    att = att / (att.sum(-1, keepdim=True) + 1e-10)
                    new_output.append(output)
                    new_attention.append(att)
                i += n_s
            sequence_output = torch.stack(new_output, dim=0)
            attention = torch.stack(new_attention, dim=0)
        return sequence_output, attention


    def __call__(self, input_ids, masks):
        sequence_output, attention = self.process_long_input(self.bert, input_ids=input_ids, attention_mask=masks)
        return sequence_output, attention 


class DocREModel(nn.Module):
    def __init__(self, bert, config) -> None:
        super().__init__()
        self.bert = bert
        self.config = config

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.encode = Encode(bert=self.bert, config=config)

    
    def forward(self, batch):
        sequence_output, attention = self.encode(input_ids=batch[0], masks=batch[1])
        return None

        

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    config = AutoConfig.from_pretrained('bert-base-cased', num_labels=97)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    
    bert = AutoModel.from_pretrained(
        'bert-base-cased',
        from_tf=False,
        config=config,
    )

    dataset = DocREDataset('../data/test.json', tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    model = DocREModel(bert, config)
    for batch in dataloader:
        x = model(batch)
        pass