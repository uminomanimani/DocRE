import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm
import math
from datetime import datetime

from docre_data import DocREDataset
from utils import  getLabelMap
from model import DocREModel, collate_fn, balanced_loss


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    config = AutoConfig.from_pretrained('bert-base-cased') #num_labels ? 
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    
    bert = AutoModel.from_pretrained(
        'bert-base-cased',
        from_tf=False,
        config=config,
    )
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f'the fucking device is {device}.')
    bert = bert.to(device)

    trainBatchSize = 32
    testBatchSize = 32

    trainDataset = DocREDataset('../data/train_annotated.json', tokenizer=tokenizer)
    trainDataloader = DataLoader(dataset=trainDataset, batch_size=trainBatchSize, collate_fn=collate_fn, shuffle=True)

    testDataset = DocREDataset('../data/dev.json', tokenizer=tokenizer)
    testDataloader = DataLoader(dataset=testDataset, batch_size=testBatchSize, collate_fn=collate_fn, shuffle=True)

    model = DocREModel(bert, config, numClass=97)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = balanced_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=5e-5)
    epochs = 200

    for epoch in range(epochs):
        total_loss = 0
        with tqdm(total=len(trainDataloader), desc=f'Epoch {epoch}/{epochs} train') as pbar_train:
            model.train()
            model = model.to(device=device)
            for batch in trainDataloader:
                input_ids, masks, entityPos = batch[0], batch[1], batch[3]
                labels, hts = batch[2], batch[4]
                labelList = []
                for i in labels:
                    for j in i:
                        labelList.append(torch.tensor(j))
                
                labels = torch.stack(labelList)
                

                input_ids = input_ids.to(device=device)
                masks = masks.to(device=device)
                labels = labels.to(device=device)
                
                optimizer.zero_grad()
                
                logits = model(input_ids, masks, entityPos, headTailPairs=hts) # (n, 97)
                # logits = torch.permute(logits, dims=(0, 2, 3, 1)) #(4, 42, 42, 97)
                
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                pbar_train.update(1)
                pass
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        with open('result.log', 'a') as f:
            print(f'{formatted_time}, Epoch : {epoch}, loss={total_loss}', file=f)
        print(f'{formatted_time}, Epoch : {epoch}, loss={total_loss}')

        with tqdm(total=len(testDataloader), desc=f'Epoch {epoch}/{epochs} test') as pbar_test:
            model.eval()
            with torch.no_grad():                
                truePos = 0
                falsePos = 0
                falseNeg = 0

                for batch in testDataloader:
                    input_ids, masks, entityPos = batch[0], batch[1], batch[3]
                    labels, hts = batch[2], batch[4]
                    labelList = []
                    for i in labels:
                        for j in i:
                            labelList.append(torch.tensor(j))
                    labels = torch.stack(labelList)

                    input_ids = input_ids.to(device=device)
                    masks = masks.to(device=device)
                    labels = labels.to(device=device)

                    pre = model(input_ids, masks, entityPos, headTailPairs=hts)

                    labels = torch.max(labels, dim=1)[1]
                    pre = torch.max(pre, dim=1)[1]
                    
                    truePos += torch.sum((labels != 0) & (labels == pre)).item()
                    falseNeg += torch.sum((labels != 0) & (labels != pre)).item()
                    falsePos += torch.sum((labels == 0) & (labels != pre)).item()
                        
                    pbar_test.update(1)
        precision = truePos / (truePos + falsePos)
        recall = truePos / (truePos + falseNeg)
        f1 = 2 * (precision * recall) / (precision + recall)
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        with open('result.log', 'a') as f:
            print(f'{formatted_time}, Epoch : {epoch}, precision = {precision * 100}%, recall = {recall * 100}%, f1 = {f1 * 100}%', file=f)
        print(f'{formatted_time}, Epoch : {epoch}, precision = {precision * 100}%, recall = {recall * 100}%, f1 = {f1 * 100}%')
                        



