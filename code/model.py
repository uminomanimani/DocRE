import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import getFeatureMap, getGraphEdges
from torch_geometric.nn import GCNConv


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


class SegmetationNet(nn.Module):
    def __init__(self, num_class : int) -> None:
        super().__init__()
        self.in_channels = 3
        self.num_class = num_class

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, self.num_class, 1)
        self.score_pool3 = nn.Conv2d(256, self.num_class, 1)
        self.score_pool4 = nn.Conv2d(512, self.num_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            self.num_class, self.num_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            self.num_class, self.num_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            self.num_class, self.num_class, 4, stride=2, bias=False)
    
    def forward(self, x):
        h = x  # (batch_size,3,42,42)
        h = self.relu1_1(self.conv1_1(h)) # (batch_size,64,240,240)
        h = self.relu1_2(self.conv1_2(h)) # (batch_size,64,240,240)
        h = self.pool1(h) # (batch_size,64,120,120)

        h = self.relu2_1(self.conv2_1(h)) # (batch_size,128,120,120)
        h = self.relu2_2(self.conv2_2(h)) # (batch_size,128,120,120)
        h = self.pool2(h) # (batch_size,128,60,60)

        h = self.relu3_1(self.conv3_1(h)) # (batch_size,256,60,60)
        h = self.relu3_2(self.conv3_2(h)) # (batch_size,256,60,60)
        h = self.relu3_3(self.conv3_3(h)) # (batch_size,256,60,60)
        h = self.pool3(h) # (batch_size,256,30,30)
        pool3 = h  # 1/8，这里pool3的形状是原图的1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h) # (batch_size,512,15,15)
        pool4 = h  # 1/16 # (batch_size,512,15,15)

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h) # (batch_size,512,8,8)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16 # (batch_size,97,6,6)

        h = self.score_pool4(pool4) # (batch_size,512,15,15)
        # s = upscore2.size()
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]] # (batch_size,79,6,6)
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

class GCNet(nn.Module):
    def __init__(self, inChannels : int, outChannels : int) -> None:
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.gcn1 = GCNConv(in_channels=self.inChannels, out_channels=2048)
        self.gcn2 = GCNConv(in_channels=2048, out_channels=self.outChannels)
    
    def forward(self, inFeatures):  # inFeatures : (batchSize, inChannels, 42, 42)
        batchFeatures = []
        for item in inFeatures: # item : (inChannels, 42, 42)
            permutedItem = torch.permute(item, dims=(1, 2, 0)) # permutedItem : (42, 42, inChannels)
            flattenItem = torch.flatten(permutedItem, end_dim=1)
            edges = getGraphEdges(size=(permutedItem.shape[0], permutedItem.shape[1]))
            edges = edges.to(item.device)
            out = self.gcn1(flattenItem, edges)
            out = self.gcn2(out, edges)
            out = torch.reshape(input=out, shape=(self.outChannels, 42, 42))
            batchFeatures.append(out)
        batchFeatures = torch.stack(batchFeatures)
        return batchFeatures



class DocREModel(nn.Module):
    def __init__(self, bert, config, numClass) -> None:
        super().__init__()
        self.bert = bert
        self.config = config
        self.numClass = numClass

        for param in self.bert.parameters():
            param.requires_grad = False
        
        self.encode = Encode(bert=self.bert, config=config)
        self.segmetation = SegmetationNet(num_class=97)
        self.gcn = GCNet(inChannels=1024, outChannels=numClass)
        # self.linear = nn.Linear(in_features=512, out_features=numClass)

    
    def forward(self, input_ids, masks, entityPos):
        sequence_output, attention = self.encode(input_ids=input_ids, masks=masks)      
        # sequence_output : (4,max_len,768) attention : (4, 12(自注意力头的数量), max_len, max_len)
        FeatureMap = getFeatureMap(sequence_output=sequence_output, entityPos=entityPos)  #这里输出的通道数为num_class
        logits = self.segmetation(FeatureMap)    # (batch_size, numClass, 42, 42)
        # logits = self.gcn(logits)
        # logits = self.linear(logits)
        return logits
