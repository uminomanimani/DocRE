import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from tqdm import tqdm

from docre_data import DocREDataset
from utils import  getLabelMap
from model import DocREModel, collate_fn


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
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bert = bert.to(device)

    dataset = DocREDataset('../data/train_annotated.json', tokenizer=tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
    model = DocREModel(bert, config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=5e-5)
    epochs = 500
    model = model.to(device=device)

    for epoch in range(epochs):
        total_loss = 0
        with tqdm(total=len(dataloader), desc=f'Epoch {epoch} train') as pbar_train:
            for batch in dataloader:
                input_ids, masks, entityPos = batch[0], batch[1], batch[3]
                labels, hts = batch[2], batch[4]
                labelMap = getLabelMap(labels=labels, hts=hts) # (4,97,42,42)
                _, labelMap = torch.max(labelMap, dim=1)

                input_ids = input_ids.to(device=device)
                masks = masks.to(device=device)
                labelMap = labelMap.to(device=device)
                
                logits = model(input_ids, masks, entityPos) # (4,97,42,42)
                loss = loss_fn(logits, labelMap)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                pbar_train.update(1)
                pass
        print(f'epoch : {epoch}, loss={total_loss}')