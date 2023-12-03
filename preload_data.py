import json
import os

import numpy as np
import torch
from tqdm import tqdm

from argument import CommandLineArgument, CommandLineFlag
from transformer import CorpusFile
from transformer.data import read_data_from_file
from transformer.model import TransformerConfig, BilingualTokenizer
from transformer.utils import extension2lang

argument_parser = CommandLineArgument()
argument_parser.define("config", "assets/config/configV1.json", str)
argument_parser.define("data_dir", "./data/raw/PhoMT/dev/", str)
argument_parser.define("save_dir", "data/preload/PhoMT/dev/", str)
argument_parser.define("chunk_size", 10240, int)
argument_parser.define("shuffle", True, bool)


def _generate_index(seed, size):
    np.random.seed(seed)
    indices = np.arange(size)
    np.random.shuffle(indices)
    return indices.tolist()


def main():
    args: CommandLineFlag = argument_parser.parse()
    
    config_file = getattr(args, "config", None)
    
    if config_file is None:
        raise ValueError("config has not been specified!")
    
    config = TransformerConfig.load(config_file)
    
    tokenizer = BilingualTokenizer(config)
    
    data_dir = getattr(args, "data_dir", None)
    
    if data_dir is None:
        raise ValueError("data_dir has not been specified!")
    
    files = os.listdir(data_dir)
    file_extensions = [x[-2:] for x in files]
    files = list(map(lambda x: data_dir + x, files))
    data = read_data_from_file(CorpusFile(url=files[0],
                                          language=extension2lang[file_extensions[0]]),
                               CorpusFile(url=files[1],
                                          language=extension2lang[file_extensions[1]]),
                               tokenizer,
                               max_length=config.max_position_embeddings)
    
    print(f"Loaded data from {data_dir}")
    
    shuffle = getattr(args, "shuffle", False)
    chunk_size = getattr(args, "chunk_size")
    print("Shuffle:", shuffle)
    if shuffle:
        print("Shuffled data")
        indices = torch.tensor(_generate_index(config.seed, len(data))).split(chunk_size)
    else:
        indices = torch.arange(len(data)).split(chunk_size)
    
    save_dir = getattr(args, "save_dir", None)
    if save_dir is None:
        raise ValueError("save_dir has not been specified!")
    metadata = {
        "size": len(data),
        "chunk_size": chunk_size,
        "seed": config.seed,
    }
    
    with open(save_dir + f"metadata.json", "w") as f:
        json.dump(metadata, f)
    
    for i, chunk_indices in enumerate(tqdm(indices)):
        input_ids = []
        target_ids = []
        for idx in chunk_indices.tolist():
            pair_data = data[idx]
            input_ids.append(pair_data[0])
            target_ids.append(pair_data[1])
        
        save_obj = (len(chunk_indices), input_ids, target_ids)
        torch.save(save_obj, save_dir + f"C{i}.pt")
    
    print(f"Saved data to {save_dir}")


if __name__ == "__main__":
    main()
