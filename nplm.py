#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nplm.py
@Time    :   2024/09/28 20:58:41
@Author  :   songxiangxiang 
@Email   :   13026117627@163.com
@description   :   xxxxxxxxx
'''
from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import logging
import os

# add logger
logging.basicConfig(level = logging.INFO, filename="./nplm/run.log", format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# x: i love ----> y:toys
# sentences = ["i love toys.",
#              "she hate game.",
#              "i love my mother!",
#              "i love my father!",
#              "he dislike learning and games!"]

# construct vocabulary
class Vocabulary:
    def __init__(self,sentences):
        
        # self.word_list = list(set(" ".join(sentences).split()))
        self.word_list = ["<PAD>"]
        for sentence in sentences:
            for word in "".join(sentence).split(" "):
                if word not in self.word_list:
                    self.word_list.append(word)
                    
        self.word2idx = {word: idx for idx,word in enumerate(self.word_list)}
        self.idx2word = {idx: word for word,idx in self.word2idx.items()}
        
    def word2idx(self):
        return self.word2idx
    
    def idx2word(self):
        return self.idx2word
    
    def __len__(self):
        return len(self.word_list)
     
     
class My_dataset(Dataset):
    
    def __init__(self,sentences):
        self.sentence = [sentence for sentence in sentences]
        self.vocabulary = Vocabulary(sentences)
        self.word2idx = self.vocabulary.word2idx
        self.idx2word = self.vocabulary.idx2word
        self.max_len = max([len("".join(sentence).split(" ")) for sentence in sentences])-1

    def __getitem__(self,index):
        
        input_x = [self.word2idx[word] for word in "".join(self.sentence[index]).split(" ")[:-1]]
        # padding
        if len(input_x) < self.max_len:
            input_x = [0]*(self.max_len-len(input_x)) + input_x
            
        input_y = [self.word2idx[word] for word in "".join(self.sentence[index]).split(" ")[-1:]]
        
        return torch.tensor(input_x),torch.tensor(input_y)
    
    def __len__(self):
        return len(self.sentence)
    
def collate_fn(batch):
    # padding by max_len
    input_x,input_y = zip(*batch)
    input_x_padding = pad_sequence(input_x,batch_first=True,padding_value=0)
    
    return input_x_padding, input_y


import torch.nn as nn

class nplm(nn.Module):
    def __init__(self, voc_size, embedding_dim, seq_len, hidden_dim):
        super(nplm,self).__init__()
        
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(voc_size,embedding_dim)
        self.linear1 = nn.Linear(seq_len*embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,voc_size)
        
    def forward(self,input_x):
        
        x = self.embedding(input_x)
        x = self.linear1(x.view(-1,self.seq_len*self.embedding_dim))
        x = torch.tanh(x)
        output = self.linear2(x)
        # print(output.size())
        
        return output

import torch.optim as optim
import numpy as np

def Train(sentences,lr,batch_size,epochs,model):
    logger.info(f"The model is {model}")
    logger.info(f"The params is {model.named_parameters}")
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=lr)
    logger.info(f"loss_function: {loss_func}")
    logger.info(f"optimizer: {optimizer}")
    
    Mydata = My_dataset(sentences)
    dataloader = DataLoader(Mydata,batch_size=batch_size,shuffle=True,drop_last=True)
    
    avg_loss = []
    min_loss = float("inf")
    for epoch in range(epochs):
        total_loss = []
        for index,(x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(x)
            # print(output.size(),y.size())
            loss = loss_func(output,y.squeeze(1))
            total_loss.append(loss)
            loss.backward()
            optimizer.step()
            
            avg_loss.append(sum(total_loss)/(index+1))
        if epoch % 1000 == 0:
            logger.info(f"Epoch:{epoch}; Loss_value:{sum(total_loss)/(index+1)}")
            # print(f"Epoch:{epoch}; Loss_value:{sum(total_loss)/(index+1)}")
        
        # calculate min_loss and save model
        if len(avg_loss) == 1:
            min_loss = avg_loss[0]
            continue
        else:
            if avg_loss[-1] <= min_loss:
                min_loss = avg_loss[-1]
                # save_model
                try:
                    torch.save(model.state_dict(), "./state_dict/model.pth")
                    # logger.info("The best model saved, the path is ./state_dict/model.pth")
                except Exception:
                    os.mkdir("./state_dict")
                    torch.save(model.state_dict(), "./state_dict/model.pth")
                    # logger.info("The best model saved, the path is ./state_dict/model.pth")
            
def Predict(input_str:list, voc, model, model_path):
    # load trained model
    try:
        model.load_state_dict(torch.load("./state_dict/model.pth"))
    except Exception:
        raise f"./state_dict/model.pth not exists,please confirm!"
    
    input_idx = []
    for sentence in input_str:
        try:
            assert type(sentence) == list
            for sen in sentence:
                sen_idx = [voc.word2idx[word] for word in "".join(sen).split(" ")]
                input_idx.append(sen_idx)
                
        except AssertionError:
            input_idx = [voc.word2idx[word] for word in "".join(sentence).split(" ")]

    input_idx_tensor = torch.tensor(input_idx)
    with torch.no_grad():
        predict = model(input_idx_tensor).max(1)[1]  # get the max value idx
        
    predict_word = [voc.idx2word[pre.item()] for pre in predict]
    
    return predict_word
        
              
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--embedding_dim", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    # train model
    sentences = ["i love toys.",
             "she hate game.",
             "i love my mother!",
             "i love my father!",
             "he dislike learning and games!"]
    voc = Vocabulary(sentences=sentences)
    voc_size = len(voc)
    
    model = nplm(voc_size=voc_size, 
                 embedding_dim=args.embedding_dim,
                 seq_len=args.seq_len,
                 hidden_dim=args.hidden_dim)

    # Train(sentences = sentences,
    #       lr = args.lr,
    #       batch_size=args.batch_size,
    #       epochs=args.epochs,
    #       model = model)
    
    res = Predict([["he dislike learning and"],
                   ["he dislike learning and"]],voc,model,"./state_dict/model.path")
    print(res)