import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from tqdm import tqdm, trange

import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
df = pd.read_csv("in_domain_train.tsv", delimiter="\t", header=None,
                 names=["sentence_source", 'label', "label_notes", "sentence"])
print(df.shape)
sentences = df.sentence.values
sentences = ["[CLS] " + sentence + "[SEP]" for sentence in sentences]
labels = df.label.values
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sentence) for sentence in sentences]
max_len = 128
input_ids = [tokenizer.convert_tokens_to_ids(token) for token in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
attention_masks = []
for seq in input_ids:
    seq_mask = [float(i>0) for i in seq]
    attention_masks.append(seq_mask)
