import os

import datasets
import torch
import torchaudio
import torchtext

import torchvision
import torchvision.transforms as transforms
from datasets import load_from_disk
from torchtext.vocab import build_vocab_from_iterator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model(dl_kwargs={"model_dir": "./model"}).to(device)
cmu_arctic = torchaudio.datasets.CMUARCTIC('./', download=True)

dataset = datasets.load_dataset('wikitext', 'wikitext-2-v1')

dataset.save_to_disk("./datasets")
dataset = load_from_disk("./datasets")

train_iter = dataset['train']
val_iter = dataset['validation']

print("data ready")
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
print("tokenizer ready")
tmp0 = (tokenizer(x['text']) for x in train_iter)
vocab = torchtext.vocab.build_vocab_from_iterator(tmp0, specials=["<unk>"])
print("vocab ready")

train_dataset = torchvision.datasets.MNIST(root='./',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
