import os

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, optim

import numpy as np

import time
import math

import matplotlib.pyplot as plt

import tqdm

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from config import Config
from network import Transformer

# sets the seed for generating random numbers.
SEED = 42
torch.manual_seed(SEED)

def get_config():
    parser = argparse.ArgumentParser(description="Transformer")
    parser.add_argument("--emb_dim", default=64, type=int)
    parser.add_argument("--ffn_dim", default=256, type=int)
    parser.add_argument("--num_attention_heads", default=4, type=int)
    parser.add_argument("--attention_drop_out", default=0.0, type=float)
    parser.add_argument("--drop_out", default=0.2, type=float)
    parser.add_argument("-max_position", default=512, type=int)
    parser.add_argument("--num_encoder_layers", default=3, type=int)
    parser.add_argument("--num_decoder_layers", default=3, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-4, type=float)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--gradient_clip", default=1, type=int)


    args = parser.parse_args()

    config = Config(
        emb_dim=args.emb_dim,
        ffn_dim=args.ffn_dim,
        num_attention_heads=args.num_attention_heads,
        attention_drop_out=args.attention_drop_out,
        drop_out=args.drop_out,
        max_position=args.max_position,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        gradient_clip=args.gradient_clip,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


    )

    return config

def prepare_data(batch_size):
    '''prepare data
    
    Args:
        batch_size (int): batch size.
        
    Returns:
        SRC (Field): source data Field class
        TRG (Field): target data Field class
        train_iterator (BucketIterator): training data iterator
        valid_iterator (BucketIterator): validation data iterator
        test_iterator (BucketIterator): test data iterator
    '''
    
    SRC = Field(tokenize = "spacy",
            tokenizer_language="de",
            init_token = '<sos>',
            eos_token = '<eos>',
            batch_first=True,
            lower = True)

    TRG = Field(tokenize = "spacy",
                tokenizer_language="en",
                init_token = '<sos>',
                eos_token = '<eos>',
                batch_first=True,
                lower = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                        fields = (SRC, TRG))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size = batch_size,
        device = device,
        shuffle=True)
    
    data_loders = dict()
    data_loders['train'] = train_iterator
    data_loders['val'] = valid_iterator
    data_loders['test'] = test_iterator
    
    return SRC, TRG, data_loders

def get_network(SRC: Field,
                TRG: Field,
                config):
    '''Get network.
    
    Args:
        SRC (Field): source data Field class.
        TRG (Field): target data Field class.
        config (Config): configuration parameters.
    
    Returns:
        model (Module): transformer model.
        criterion (CrossEntropyLoss): loss function. 
        optimizer (Adam): optimizer.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer(SRC, TRG, config).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index = TRG.vocab.stoi['<pad>'])
    optimizer = optim.Adam(model.parameters(),lr = config.learning_rate)
    
    return model, criterion, optimizer

def print_model_info(model: nn.Module,
                     optimizer: torch.optim):

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())
    
    print('\n\n')

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

def save_model(model: nn.Module):
    if not os.path.isdir('ckpt'):
        os.mkdir('ckpt')
    now = time.localtime()
    path = 'ckpt/' + f"{now.tm_year}-{now.tm_mon}-{now.tm_mday}_{now.tm_hour}_{now.tm_min}_{now.tm_sec}" + '.pt'

    torch.save(model.state_dict(), path)

def train(model: nn.Module,
          data_loders: dict,
          criterion,
          optimizer,
          config):
    '''Training model
    
    Args:
        model (nn.Module): transformer model.
        data_loders (dict): training/validation data iterator.
        criterion : loss function. 
        optimizer : optimizer.
        config (Config): configuration parameters.
    '''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    
    print_loss_every = 1
    for epoch in range(config.n_epochs): 
        for phase in ['train', 'val']:
            
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            loss_val_sum = 0
            
            for batch in data_loders[phase]:
                
                optimizer.zero_grad()
 
                src = batch.src.to(device)
                trg = batch.trg.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    output, _, _ = model(src, trg)

                    output = output[:,:-1,:].reshape(-1, output.shape[-1])
                    trg = trg[:,1:].reshape(-1)

                    loss = criterion(output, trg)
                    
                    if phase == 'train':
                        loss.backward()
                    # gradient clipping
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                        optimizer.step()

                loss_val_sum += loss

            if ((epoch % print_loss_every) == 0) or (epoch == (config.n_epochs - 1)):
                loss_val_avg = loss_val_sum / len(data_loders[phase])
                print(
                    f"epoch:[{epoch+1}/{config.n_epochs}] {phase} cost:[{loss_val_avg:.3f}]"
                )
    print('training done!!')
    save_model(model)
if __name__ == "__main__":
    print("PyTorch version:[%s]." % (torch.__version__))

    config = get_config()
    print("This code use [%s]." % (config.device))

    SRC, TRG, data_loders = prepare_data(config.batch_size)
    model, criterion, optimizer = get_network(SRC, TRG, config)
    print_model_info(model, optimizer)

    train(model, data_loders, criterion, optimizer, config)
