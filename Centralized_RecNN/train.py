import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from NeuMF import NCF
from utils import HMSaleTrainDataLoader, sparse_batch_collate, sparse_to_tensor, binary_acc

pd.options.mode.chained_assignment = None
dataDir = Path.cwd().parent.parent.parent/'backup/HM_data'
np.random.seed(66)

transactions = pd.read_csv(dataDir/'transactions.csv')
# articles = pd.read_csv(dataDir/'articles.csv')
# customers = pd.read_csv(dataDir/'customers.csv')

# calculate customer interactions in transaction data
# drop customers that only contain few interactions
transactions["interaction"] = 1
transactions_temp = transactions.drop_duplicates(subset=["customer_id", "article_id"])
comb_transactions = transactions_temp[["customer_id", "interaction"]].groupby(by=["customer_id"], sort=False, as_index=False).sum(["interaction"])
comb_transactions = comb_transactions.loc[comb_transactions.interaction >= 5]

# randomly select part of the transaction data
rand_userIds = np.random.choice(comb_transactions.customer_id.unique(), 
                                size=int(len(comb_transactions['customer_id'].unique())*0.003), 
                                replace=False)

transactions = transactions.loc[transactions['customer_id'].isin(rand_userIds)]

print('There are {} rows of data from {} users (users with suffication data)'.format(len(transactions), len(rand_userIds)))

transactions.drop_duplicates(subset=["customer_id", "article_id"], keep="first", inplace=True)

# training set and test set
transactions['rank_latest'] = transactions.groupby(['customer_id'])['t_dat'].rank(method='first', ascending=False)

train_transactions = transactions[transactions['rank_latest'] != 1]
test_transactions = transactions[transactions['rank_latest'] == 1]

# drop articles that do not exist in training set
test_product_list = list(set(test_transactions.article_id.unique()) & set(train_transactions.article_id.unique()))
test_transactions = test_transactions.loc[test_transactions['article_id'].isin(test_product_list)]

"""reindex"""
# map string type customer_id to int type
customer_mapper = {}
customer_keys = train_transactions.customer_id.unique()
customer_values = list(range(len(train_transactions.customer_id.unique())))
customer_mapper.update(zip(customer_keys, customer_values))

# map string type article_id to int type
product_mapper = {}
product_keys = train_transactions.article_id.unique()
product_values = list(range(len(train_transactions.article_id.unique())))
product_mapper.update(zip(product_keys, product_values))

train_transactions["customer_id"] = train_transactions["customer_id"].map(customer_mapper)
train_transactions["article_id"] = train_transactions["article_id"].map(product_mapper)
test_transactions["customer_id"] = test_transactions["customer_id"].map(customer_mapper)
test_transactions["article_id"] = test_transactions["article_id"].map(product_mapper)

# get a list of all articles id
all_products_id = train_transactions["article_id"].unique()

# set up hyper-parameters
learning_rate = 0.01
epoch = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# set up dataset for training
num_users = len(train_transactions.customer_id.unique())
print("num_users:", num_users)
num_items = len(all_products_id)
print("num_items:", num_items)
train_data = HMSaleTrainDataLoader(train_transactions, all_products_id)
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)

# initiate model for training
model = NCF(num_users=num_users, num_items=num_items)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
model.train()
best_acc = 0.0
for e in tqdm(range(epoch)):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for customer_batch, product_batch, label_batch in train_loader:
        customer_batch, product_batch, label_batch = customer_batch.to(device), product_batch.to(device), label_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(customer_batch, product_batch)
        loss = loss_fn(y_pred, label_batch)
        acc = binary_acc(y_pred, label_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        cur_acc = epoch_acc/len(train_loader)
    if cur_acc > best_acc:
        best_acc = cur_acc
        torch.save(model.state_dict(), 'best_model.pt')
    print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.6f} | Acc: {epoch_acc/len(train_loader):.4f}')

print(f'\nBest Accuracy: {best_acc:.3f}')