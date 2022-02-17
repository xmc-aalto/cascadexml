import torch
import torch.nn.functional as f
import numpy as np

def word_pool(words, tfidfs, out_shape):
    assert out_shape <= words.shape[0]
    
    chunks = np.array_split(np.arange(words.shape[0]), out_shape)
    new_words = [(words[c] * tfidf[c]).sum(0) for c in chunks]

    return torch.stack(new_words)