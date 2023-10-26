import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

from load_data import ReadDataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class DocREDataset(Dataset):
    def __init__(self, file : str, tokenizer):
        self.tokenizer = tokenizer
        self.readDataset = ReadDataset('docred', self.tokenizer, max_seq_Length=1024)

        self.features = self.readDataset.read(file_in=file)
        pass
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index]

if __name__ == '__main__':
    d = DocREDataset('../data/test.json')
    
    pass