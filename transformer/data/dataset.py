import json
import os.path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..dataclasses_utils import CorpusFile
from ..model.tokenization import BilingualTokenizer


def read_data_from_file(file1: CorpusFile,
                        file2: CorpusFile,
                        tokenizer: BilingualTokenizer,
                        max_length: int = 128):
    buffered_reader_1 = open(file1.url, "r", encoding="utf-8")
    buffered_reader_2 = open(file2.url, "r", encoding="utf-8")
    
    data = []
    
    for _, (br1_line, br2_line) in enumerate(tqdm(zip(buffered_reader_1, buffered_reader_2))):
        br1_line = br1_line.strip()
        br2_line = br2_line.strip()
        
        ids_1 = tokenizer.encode(text=br1_line, bos_id=file1.language.value)
        ids_2 = tokenizer.encode(text=br2_line, bos_id=file2.language.value)
        
        if len(ids_1) > max_length or len(ids_2) > max_length:
            continue
        
        data.extend([(ids_1, ids_2), (ids_2, ids_1)])
    
    buffered_reader_1.close()
    buffered_reader_2.close()
    
    return data


class PreLoadDataset(Dataset):
    def __init__(self,
                 chunk_files: list[str],
                 dataset_size: int,
                 chunk_size: int):
        self.chunk_files = chunk_files
        self.dataset_size = dataset_size
        self.chunk_size = chunk_size
        self.chunk_idx = 0
        self.chunk = torch.load(chunk_files[self.chunk_idx])
        
        n_chunks = int((dataset_size - 1) // chunk_size) + 1
        
        self.first_chunk_index = [chunk_size*i for i in range(n_chunks)]
    
    def __len__(self):
        return self.dataset_size
    
    def __getitem__(self, idx):
        low = self.first_chunk_index[self.chunk_idx]
        high = low + self.chunk[0]
        
        if low <= idx < high:
            idx = idx - low
        else:
            self.chunk_idx = idx // self.chunk_size
            self.chunk = torch.load(self.chunk_files[self.chunk_idx])
            low = self.first_chunk_index[self.chunk_idx]
            idx = idx - low
            
        return self.chunk[1][idx], self.chunk[2][idx]


def load_dataset(data_dir, config):
    with open(data_dir + "metadata.json", "r") as f:
        data_config = json.load(f)
    
    assert data_config["seed"] == config.seed
    
    dataset_size = data_config["size"]
    chunk_size = data_config["chunk_size"]
    
    chunk_files = [file for file in os.listdir(data_dir) if file.endswith(".pt")]
    chunk_files = [data_dir + file for file in sorted(chunk_files, key=lambda x: int(x[1:-3]))]
    
    return PreLoadDataset(chunk_files, dataset_size, chunk_size)
