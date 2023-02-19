import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from NeuMF import NCF
from utils import HMSaleTrainDataLoader, binary_acc

pd.options.mode.chained_assignment = None
dataDir = Path.cwd().parent.parent.parent/'backup/HM_data'
np.random.seed(66)

articles_usecols = ["article_id", "product_group_name", "colour_group_name", "index_name"]
customers_usecols = ["customer_id", "club_member_status", "age"]

transactions = pd.read_csv(dataDir/'transactions.csv')
articles = pd.read_csv(dataDir/'articles.csv', usecols=articles_usecols)
customers = pd.read_csv(dataDir/'customers.csv', usecols=customers_usecols)

""" Preprocessing on articles and customers data
    Articles: product_group_name, colour_group_name, index_name
    Customers: club_member_status, age
"""

articles = articles.loc[articles.product_group_name != "Unknown"]
articles = articles.loc[articles.colour_group_name != "Unknown"]
customers.dropna(axis=0, how='any', subset=["club_member_status", "age"], inplace=True)
# filter out transactions data with articles and customers ID
transactions = transactions.loc[transactions['article_id'].isin(articles.article_id.unique())]
transactions = transactions.loc[transactions['customer_id'].isin(customers.customer_id.unique())]

# calculate customer interactions in transaction data
# drop customers that only contain few interactions
transactions["interaction"] = 1
transactions_temp = transactions.drop_duplicates(subset=["customer_id", "article_id"])
comb_transactions = transactions_temp[["customer_id", "interaction"]].groupby(by=["customer_id"], sort=False, as_index=False).sum(["interaction"])
comb_transactions = comb_transactions.loc[comb_transactions.interaction >= 5]

# randomly select part of the transaction data
rand_userIds = np.random.choice(comb_transactions.customer_id.unique(), 
                                size=int(len(comb_transactions['customer_id'].unique())*0.001), 
                                replace=False)

transactions = transactions.loc[transactions['customer_id'].isin(rand_userIds)]

print('There are {} rows of data from {} users (users with suffication data)'.format(len(transactions), len(rand_userIds)))

transactions.drop_duplicates(subset=["customer_id", "article_id"], keep="first", inplace=True)

# merge transaction data with article and customer data
transactions = transactions.merge(customers, how='left', left_on=["customer_id"], right_on=["customer_id"])
transactions = transactions.merge(articles, how='left', left_on=["article_id"], right_on=["article_id"])

# binarize categorical features club_member_status and sales_channel_id in transaction data
lb = LabelBinarizer()
transactions.sales_channel_id = lb.fit_transform(transactions.sales_channel_id)
transactions.club_member_status = lb.fit_transform(transactions.club_member_status)

# standardize numerical features age and price in transaction data
std = StandardScaler()
transactions[["price", "age"]] = std.fit_transform(transactions[["price", "age"]])

# training set and test set
transactions['rank_latest'] = transactions.groupby(['customer_id'])['t_dat'].rank(method='first', ascending=False)

train_transactions = transactions[transactions['rank_latest'] != 1]
test_transactions = transactions[transactions['rank_latest'] == 1]

# drop articles that do not exist in training set
test_product_list = list(set(test_transactions.article_id.unique()) & set(train_transactions.article_id.unique()))
test_transactions = test_transactions.loc[test_transactions['article_id'].isin(test_product_list)]

# drop columns that we no longer need
drop_cols = ["t_dat", "interaction", "rank_latest"]
train_transactions.drop(labels=drop_cols, axis=1, inplace=True)
test_transactions.drop(labels=drop_cols, axis=1, inplace=True)

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

# map color_group_name to int type
color_mapper = {}
color_keys = train_transactions.colour_group_name.unique()
color_values = list(range(len(train_transactions.colour_group_name.unique())))
color_mapper.update(zip(color_keys, color_values))

# map color_group_name to int type
product_group_mapper = {}
product_group_keys = train_transactions.product_group_name.unique()
product_group_values = list(range(len(train_transactions.product_group_name.unique())))
product_group_mapper.update(zip(product_group_keys, product_group_values))

# map index_name to int type
index_name_mapper = {}
index_name_keys = train_transactions.index_name.unique()
index_name_values = list(range(len(train_transactions.index_name.unique())))
index_name_mapper.update(zip(index_name_keys, index_name_values))

# reindex categorical features based on feature mappers
train_transactions["customer_id"] = train_transactions["customer_id"].map(customer_mapper)
train_transactions["article_id"] = train_transactions["article_id"].map(product_mapper)
train_transactions["colour_group_name"] = train_transactions["colour_group_name"].map(color_mapper)
train_transactions["product_group_name"] = train_transactions["product_group_name"].map(product_group_mapper)
train_transactions["index_name"] = train_transactions["index_name"].map(index_name_mapper)
test_transactions["customer_id"] = test_transactions["customer_id"].map(customer_mapper)
test_transactions["article_id"] = test_transactions["article_id"].map(product_mapper)
test_transactions["colour_group_name"] = test_transactions["colour_group_name"].map(color_mapper)
test_transactions["product_group_name"] = test_transactions["product_group_name"].map(product_group_mapper)
test_transactions["index_name"] = test_transactions["index_name"].map(index_name_mapper)

# get a list of all articles id
all_products_id = train_transactions["article_id"].unique()

train_transactions.head()

# set up hyper-parameters
learning_rate = 0.005
epoch = 40
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)

# set up dataset for training
num_users = len(train_transactions.customer_id.unique())
print("num_users:", num_users)
num_items = len(all_products_id)
print("num_items:", num_items)
num_product_groups = len(train_transactions.product_group_name.unique())
print("num_product_groups:", num_product_groups)
num_color_groups = len(train_transactions.colour_group_name.unique())
print("num_color_groups:", num_color_groups)
num_index_name = len(train_transactions.index_name.unique())
print("num_index_name:", num_index_name)
train_data = HMSaleTrainDataLoader(train_transactions, all_products_id)
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True)

# initiate model for training
model = NCF(num_users=num_users, num_items=num_items, num_product_groups=num_product_groups, num_color_groups=num_color_groups, num_index_name=num_index_name)
model.to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
model.train()
best_acc = 0.0
for e in tqdm(range(epoch)):
    epoch_loss = 0.0
    epoch_acc = 0.0
    for customer_batch, product_batch, prices_batch, sales_channels_batch, club_status_batch, age_groups_batch, product_groups_batch, color_groups_batch, index_name_batch, label_batch in train_loader:
        customer_batch, product_batch, prices_batch, sales_channels_batch, club_status_batch, \
        age_groups_batch, product_groups_batch, color_groups_batch, index_name_batch, label_batch \
        = customer_batch.to(device), product_batch.to(device), prices_batch.to(device), \
        sales_channels_batch.to(device), club_status_batch.to(device), age_groups_batch.to(device), \
        product_groups_batch.to(device), color_groups_batch.to(device), index_name_batch.to(device), label_batch.to(device), 
        optimizer.zero_grad()
        y_pred = model(customer_batch, product_batch, prices_batch.float(), sales_channels_batch.float(), club_status_batch.float(), age_groups_batch.float(), product_groups_batch, color_groups_batch, index_name_batch)
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

print(f'\nTraining Best Accuracy: {best_acc:.3f}')


""" Model Testing """

model = NCF(num_users=num_users, num_items=num_items, num_product_groups=num_product_groups, num_color_groups=num_color_groups, num_index_name=num_index_name)
test_data = HMSaleTrainDataLoader(test_transactions, all_products_id)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

model.load_state_dict(torch.load('best_model.pt'))
model.to(device)

model.eval()
test_acc = 0.0
with torch.no_grad():
    for customer_batch, product_batch, prices_batch, sales_channels_batch, club_status_batch, age_groups_batch, product_groups_batch, color_groups_batch, index_name_batch, label_batch in test_dataloader:
        customer_batch, product_batch, prices_batch, sales_channels_batch, club_status_batch, \
        age_groups_batch, product_groups_batch, color_groups_batch, index_name_batch, label_batch \
        = customer_batch.to(device), product_batch.to(device), prices_batch.to(device), \
        sales_channels_batch.to(device), club_status_batch.to(device), age_groups_batch.to(device), \
        product_groups_batch.to(device), color_groups_batch.to(device), index_name_batch.to(device), label_batch.to(device), 
        
        y_pred = model(customer_batch, product_batch, prices_batch.float(), sales_channels_batch.float(), club_status_batch.float(), age_groups_batch.float(), product_groups_batch, color_groups_batch, index_name_batch)
        acc = binary_acc(y_pred, label_batch)
        test_acc += acc.item()
    test_acc = test_acc/len(test_dataloader)

print(f'\nClassification Accuracy on test set: {test_acc:.3f}')