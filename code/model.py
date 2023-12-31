import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import getFeatureMap, getGraphEdges, getEntityEmbeddings
from torch_geometric.nn import GCNConv

from Segmentation import FCN8s as SegmentationNet


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


class GCNet(nn.Module):
    def __init__(self, inChannels : int, outChannels : int) -> None:
        super().__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.gcn1 = GCNConv(in_channels=self.inChannels, out_channels=128)
        self.gcn2 = GCNConv(in_channels=128, out_channels=self.outChannels)
        # self.weights = nn.Parameter()

    def forward(self, inFeatures):  # inFeatures : (batchSize, inChannels, 42, 42)
        batchFeatures = []
        for item in inFeatures: # item : (inChannels, 42, 42)
            permutedItem = torch.permute(item, dims=(1, 2, 0)).contiguous() # permutedItem : (42, 42, inChannels)
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
        self.featureDim = 512
        self.headLinear = nn.Linear(in_features=768, out_features=self.featureDim)
        self.tailLinear = nn.Linear(in_features=768, out_features=self.featureDim)

        self.bilinear = nn.Bilinear(in1_features=self.featureDim, in2_features=self.featureDim, out_features=numClass)

        # for param in self.bert.parameters():
        #     param.requires_grad = True

        self.encode = Encode(bert=self.bert, config=config)
        self.segmetation = SegmentationNet(inChannels=3, num_class=self.featureDim // 2)
        self.gcn = GCNet(inChannels=3, outChannels=self.featureDim // 2)


    def forward(self, input_ids, masks, entityPos, headTailPairs):
        sequence_output, attention = self.encode(input_ids=input_ids, masks=masks)
        # sequence_output : (4,max_len,768) attention : (4, 12(自注意力头的数量), max_len, max_len)
        batchEntityEmb = getEntityEmbeddings(sequence_output=sequence_output, entityPos=entityPos)
        FeatureMap = getFeatureMap(sequence_output=sequence_output, batchEntityEmbeddings=batchEntityEmb)  #这里输出的通道数为num_class
        # FeatureMap : (batchsize, 3, 42, 42)
        s_output = self.segmetation(FeatureMap)    # (batch_size, featureDim / 2, 42, 42)
        g_output = self.gcn(FeatureMap)     # (batch_size, featureDim / 2, 42, 42)
        output = torch.cat((s_output, g_output), dim=1)
        output = torch.permute(output, dims=(0, 2, 3, 1)).contiguous() #(batchsize, 42, 42, self.featureDim)

        # return output

        headTailPairEmb = []
        headEmbs = []
        tailEmbs = []
        for i in range(len(headTailPairs)):
            headTailPair = headTailPairs[i]
            entityEmb = batchEntityEmb[i] # 这个是实体的embedding，按照出现的顺序排列的
            for (h, t) in headTailPair:
                headTailPairEmb.append(output[i][h, t])
                headEmbs.append(entityEmb[h])
                tailEmbs.append(entityEmb[t])
        headTailPairEmb = torch.stack(headTailPairEmb, dim=0)
        headEmbs = torch.stack(headEmbs, dim=0)
        tailEmbs = torch.stack(tailEmbs, dim=0)

        zs = torch.tanh(self.headLinear(headEmbs) + headTailPairEmb)
        zo = torch.tanh(self.tailLinear(tailEmbs) + headTailPairEmb)

        # del headTailPairEmb, headEmbs, tailEmbs, output, sequence_output

        logits = self.bilinear(zs, zo)

        return logits


def multilabel_categorical_crossentropy(y_true, y_pred):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e30
    y_pred_pos = y_pred - (1 - y_true) * 1e30
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros],dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros),dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
    pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
    return neg_loss + pos_loss


class balanced_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        loss = multilabel_categorical_crossentropy(labels,logits)
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = torch.zeros_like(logits[..., :1]) #(1310, 1)
        output = torch.zeros_like(logits).to(logits) #(1310,97)
        mask = (logits > th_logit) # mask是logits中大于0的那些值，0：大于0；1：小于0，(1310, 97)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1) #获得logits的97个值中前num_labels个值,降序排列 (1310, num_labels)
            top_v = top_v[:, -1] #获得最小那个, (1310)
            mask = (logits >= top_v.unsqueeze(1)) & mask # logits >= top_v.unsqueeze(1) 这个是在求哪些是大于前num_labels个值中最小的那个值
        output[mask] = 1.0
        output[:, 0] = (output[:,1:].sum(1) == 0.).to(logits)

        return output #其中只有最大的num_labels个元素对应的位置被设置为1，其余位置为0。
