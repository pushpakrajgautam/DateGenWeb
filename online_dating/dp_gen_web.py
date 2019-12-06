import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.autograd import Variable
from nltk.translate import bleu_score
from html2text import html2text
from django.conf import settings

from .configs import cfg
from .model import *

# Model and Data
model = None
torch_device = None
X_train = None
y_train = None 
X_valid = None
y_valid = None

# Helpers

def print_unique_attributes(data):
    for c in get_in_columns():
        print(set(data[c]))

def get_in_columns():
    return ['age', 'body_type', 'drinks', 'education', 'job', 'orientation', 
            'pets', 'sex', 'smokes', 'status']

def get_out_column():
    return "essay0"

def toword(num):
    idxw = cfg['idx_to_word']
    return idxw[int(num)]

def save_to_file(outputs, fname):
    with open(fname, "a") as myfile:
        for n in outputs:
            myfile.write(n + "\n")

# Core

def load_data(fname):
    """
    From the csv file given by fname, return a pandas DataFrame
    """
    df = pd.read_csv(fname)
    df.dropna(subset=[get_out_column()], inplace=True)
    print(df.shape)
    return df 

def process_train_data(data):
    """
    Setup one hot encoding for input data and return attributes and text
    """
    # Setup input data
    in_columns = get_in_columns()
    df = data[in_columns]
    # One-hot encode columns and store dummies in config
    dummies = []
    ds = dict() # dummy store - list of dicts to remember one-hots of attributes
    for col in in_columns:
        if col != "age":
            dummy = pd.get_dummies(df[col])
            xs = dict()
            for x in zip(dummy.values, df[col]):
                xs[x[1]] = x[0]
            ds[col] = xs
            dummies.append(dummy)
    cfg['dummies'] = ds
    # Limit age to 70
    def limit_age(age):
        return age if age <= 70 else 70
    age = df["age"].apply(limit_age)
    # Join attribute columns
    dummies.append(pd.DataFrame(age))
    attr = pd.concat(dummies, axis=1)
    
    # Setup output data
    cw = set()
    def one_line(text):
        hlist = html2text(text.split("<br />")[0].strip(".")).lower().split(" ")
        htext = [h for h in hlist if h.isalpha()]
        for h in htext:
            if h not in cw:
                cw.add(h)
        return " ".join(htext)
    oc = get_out_column()
    doc = data[oc].apply(one_line)
    doc = doc[doc != ""]

    # Set up one-hot for words
    widx = dict()
    idxw = dict()
    for i, w in enumerate(cw):
        widx[w] = i
        idxw[i] = w
    widx['<SOS>'] = len(cw)
    idxw[len(cw)] = '<SOS>'
    widx['<EOS>'] = len(cw) + 1
    idxw[len(cw)+1] = '<EOS>'
    cfg['vocab_size'] = len(cw) + 2
    cfg['word_to_idx'] = widx
    cfg['idx_to_word'] = idxw

    # Return input and output data
    return doc.as_matrix(), attr.as_matrix()

def process_test_input(df):
    # Setup input data
    in_columns = get_in_columns()
    # One-hot encode columns
    dummies = []
    for col in in_columns:
        if col != "age":
            dummies.append(pd.get_dummies(df[col]))
    # Limit age to 70
    def limit_age(age):
        return age if age <= 70 else 70
    age = df["age"].apply(limit_age)
    # Join attribute columns
    dummies.append(pd.DataFrame(age))
    attr = pd.concat(dummies, axis=1)
    return attr.as_matrix()

def train_valid_split(data, labels):
    """
    Split data into tran and valid 80:20
    """
    split = (int)(len(data) * 0.8)
    return data[0:split], labels[0:split], data[split:], labels[split:] 

def batch_text_oh(text, labels):
    vs = cfg['vocab_size']
    widx = cfg['word_to_idx']

    max_len = 0
    for a in text:
        if type(a) is float and np.isnan(a):
            continue
        words = a.split(" ")
        if len(words) > max_len:
            max_len = len(words)
    if max_len > cfg['max_words']:
        max_len = cfg['max_words']
    max_len +=2
    arr = np.zeros((len(text), max_len, vs))
    lab = np.zeros((len(text), max_len, 102))
    for i in range(len(text)):
        arr[i, 0, vs-2] = 1
        lab[i, 0] = labels[i]
        for j in range(max_len):
            lab[i, j] = labels[i]
            words = text[i].split(" ")
            words_len = len(words) if len(words) < cfg['max_words'] else cfg['max_words']
            if j < words_len:
                arr[i, j + 1, widx[words[j]]] = 1
            else:
                arr[i, j, vs-1] = 1
    return arr, lab, max_len

def train(model, X_train, y_train, X_valid, y_valid, cfg):
    model.zero_grad()
    model.train(mode=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['L2_penalty'])

    for epoch in range(0, cfg['epochs']):
        rng = np.random.get_state()
        np.random.shuffle(X_train)
        np.random.set_state(rng)
        np.random.shuffle(y_train)

        for batch in range((int)(len(X_train)/cfg['batch_size'])):
            model.train(mode=True)
            model.zero_grad()
            optimizer.zero_grad()
            print("Batch: " + str(batch))
            total_loss = 0
            x_t = X_train[batch*cfg['batch_size']:(batch + 1) * cfg['batch_size']]
            y_t = y_train[batch*cfg['batch_size']:(batch + 1) * cfg['batch_size']] 
            
            oh, oh_l, max_len = batch_text_oh(x_t, y_t)
            model.hidden_state = model.zero_hidden()
            len_x_t = len(x_t)
            del x_t, y_t

            train_input = np.concatenate((oh[:, :-1, :], oh_l[:, :-1, :]), axis=2)
            train_truth = oh[:, 1:, :]
            del oh, oh_l

            train_input = np.transpose(train_input, (1, 0, 2))
            train_truth = np.transpose(train_truth, (1, 0, 2))
            print(train_input.shape)

            result = model(torch.from_numpy(train_input).float().to(torch_device))
            tensor = torch.from_numpy(train_truth).float().to(torch_device)
            f = np.vectorize(toword)
            del train_input, train_truth

            loss = 0
            for a in range(len_x_t):
                loss += criterion(result[:, a, :], tensor[:, a, :].long().argmax(dim=1))
            loss.backward()
            optimizer.step()

            print("Total Loss: " + str(loss / cfg['batch_size']))
            del max_len, result, tensor, loss

            if batch % 200 == 0 and batch != 0:
                print("Batch: " + str(batch), flush=True)
                model.eval()
                with torch.no_grad():
                    loss = 0
                    for batch1 in range(5):
                        x_t = X_valid[batch1*cfg['batch_size']:(batch1 + 1) * cfg['batch_size']]
                        y_t = y_valid[batch1*cfg['batch_size']:(batch1 + 1) * cfg['batch_size']] 
                        len_x_t = len(x_t)
                        oh, oh_l, max_len = batch_text_oh(x_t, y_t)
                        del x_t, y_t

                        model.hidden_state = model.zero_hidden()

                        train_input = np.concatenate((oh[:, :-1, :], oh_l[:, :-1, :]), axis=2)
                        train_input = np.transpose(train_input, (1, 0, 2))
                        train_truth = oh[:, 1:, :]
                        train_truth = np.transpose(train_truth, (1, 0, 2))
                        del oh, oh_l

                        result = model(torch.from_numpy(train_input).float().to(torch_device))
                        tensor = torch.from_numpy(train_truth).float().to(torch_device)
                        del train_input, train_truth

                        for a in range(len_x_t):
                            loss += criterion(result[:, a, :], tensor[:, a, :].long().argmax(dim=1)) / (cfg['batch_size'])

                        del max_len, result, tensor

                    loss = loss / 5 
                    print("Validation Loss: " + str(loss))
                    del loss

        torch.save(model.state_dict(), str(epoch) + "LSTM.txt")
        print("Model saved to: " + str(epoch) + "LSTM.txt")

def validate(model, X_valid, y_valid, cfg):
    bs = cfg['batch_size']
    vs = cfg['vocab_size']
    for a in range(20):
        starts = np.zeros((bs, vs))
        output_tensor = np.zeros((1, bs))
        for i in range(len(starts)):
            starts[i, 128] = 1
            output_tensor[0, i] = 128

        metad = y_valid[a*bs:(a + 1)*bs, :]
        features = np.concatenate((starts, metad), axis=1)
        features = np.expand_dims(features, axis=0)
        model.hidden_state = model.zero_hidden()

        for k in range(cfg['max_len']):
            with torch.no_grad():
                result = model(torch.from_numpy(features).float().to(torch_device))
                result = func.softmax(torch.div(result, cfg['gen_temp']), dim=2)
                result = torch.distributions.one_hot_categorical.OneHotCategorical(result)
                result = result.sample()
                output_tensor = np.concatenate((output_tensor, result.argmax(dim=2)), axis=0) 
                features = np.concatenate((result, np.expand_dims(metad, axis=0)), axis=2)

        f = np.vectorize(toword)
        output_tensor = f(output_tensor).T
        output_tensor = [' '.join(row) for row in output_tensor]
        split = [[[x.strip() for x in ss.strip().split(" ")] for ss in s.split(".")] for s in output_tensor]
        split_ref = [[[x.strip() for x in ss.strip().split(" ")] for ss in s.split(".")] for s in X_valid]

        score = 0
        for b in range(len(split)):
            arr = split_ref[b]
            arr = [[s] for s in arr]
            min_len = 0
            if len(arr) < len(split[b]):
                min_len = len(arr)
            else:
                min_len = len(split[b])
            score += bleu_score.corpus_bleu(arr[0:min_len], split[b][0:min_len])

    print("BLEU Score: " + str(score / (bs*20)))

def generate_to_file(model, X_test, cfg):
    bs = cfg['batch_size']
    vs = cfg['vocab_size']
    for a in range((int)(len(X_test)/bs)):
        starts = np.zeros((bs, vs))
        output_tensor = np.zeros((1, bs))
        for i in range(len(starts)):
            starts[i, 128] = 1
            output_tensor[0, i] = 128

        metad = X_test[a*bs:(a + 1)*bs, :]
        features = np.concatenate((starts, metad), axis=1)
        features = np.expand_dims(features, axis=0)
        model.hidden_state = model.zero_hidden()

        for k in range(cfg['max_len']):
            with torch.no_grad():
                result = model(torch.from_numpy(features).float().to(torch_device))
                result = func.softmax(torch.div(result, cfg['gen_temp']), dim=2)
                result = torch.distributions.one_hot_categorical.OneHotCategorical(result)
                result = result.sample()
                output_tensor = np.concatenate((output_tensor, result.argmax(dim=2)), axis=0) 
                features = np.concatenate((result, np.expand_dims(metad, axis=0)), axis=2)

        f = np.vectorize(toword)
        output_tensor = f(output_tensor).T
        output_tensor = [' '.join(row) for row in output_tensor]
        save_to_file(output_tensor, out_fname) 

def generate_profile(model, X_test, cfg):
    bs = cfg['batch_size']
    vs = cfg['vocab_size']
    starts = np.zeros((bs, vs))
    output_tensor = np.zeros((1, bs))
    for i in range(len(starts)):
        starts[i, 128] = 1
        output_tensor[0, i] = 128

    metad = np.array([X_test])
    features = np.concatenate((starts, metad), axis=1)
    features = np.expand_dims(features, axis=0)
    model.hidden_state = model.zero_hidden()

    for k in range(cfg['max_len']):
        with torch.no_grad():
            result = model(torch.from_numpy(features).float().to(torch_device))
            result = func.softmax(torch.div(result, cfg['gen_temp']), dim=2)
            result = torch.distributions.one_hot_categorical.OneHotCategorical(result)
            result = result.sample()
            output_tensor = np.concatenate((output_tensor, result.argmax(dim=2)), axis=0) 
            features = np.concatenate((result, np.expand_dims(metad, axis=0)), axis=2)

    f = np.vectorize(toword)
    output_tensor = f(output_tensor).T
    profile = [' '.join(row) for row in output_tensor]
    res = []
    for p in profile[0].split(' '):
        if p == "<EOS>":
            break
        res.append(p)
    return ' '.join(res)

def get_profile(input):
    ds = cfg["dummies"]
    in_columns = get_in_columns()
    pds = []
    for col in in_columns:
        if col != "age":
            pds.append(pd.Series(ds[col][input[col]]))
    pds.append(pd.Series([int(input["age"])]))
    attr = pd.concat(pds, axis=0)
    cfg["gen_temp"] = float(input["temp"])
    print(attr.as_matrix())
    profile = generate_profile(model, attr.as_matrix(), cfg)
    return profile

def setup():
    global model
    global torch_device
    global X_train
    global y_train
    global X_valid
    global y_valid
    if cfg['cuda'] and torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')
    cfg['batch_size'] = 1
    data = load_data('profiles.csv')
    out_fname = "gen_profiles.txt"
    train_data, train_labels = process_train_data(data)
    X_train, y_train, X_valid, y_valid = train_valid_split(train_data, train_labels)
    model = baselineLSTM(cfg)
    model.load_state_dict(torch.load("./4LSTM.txt", map_location=torch_device))