from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import numpy as np

np.random.seed(666)

class HMSaleTrainDataLoader(Dataset):
    """HMSaleTrainDataLoader Training set of HM sales data

    Args:
        transactions (pd.DataFrame): Dataframe of transaction records
        all_products_id (list): A list contains all product ids
    """
    def __init__(self, transactions, all_products_id):
        self.customers, self.products, self.prices, self.sales_channels, \
        self.club_status, self.age_groups, self.product_groups, self.color_groups, \
        self.index_name, self.labels = self.get_dataset(transactions, all_products_id)

    def __len__(self):
        return len(self.customers)
    
    def __getitem__(self, idx):
        return self.customers[idx], self.products[idx], self.prices[idx], self.sales_channels[idx], self.club_status[idx], \
               self.age_groups[idx], self.product_groups[idx], self.color_groups[idx], self.index_name[idx], self.labels[idx]
    
    def get_dataset(self, transactions, all_products_id):
        customers, products, prices, sales_channels, club_status, age_groups, product_groups, color_groups, index_name, labels  = [], [], [], [], [], [], [], [], [], []
        customer_product_set = set(zip(transactions["customer_id"], transactions["article_id"], 
                                       transactions["price"], transactions["sales_channel_id"], 
                                       transactions["club_member_status"], transactions["age"], 
                                       transactions["product_group_name"], transactions["colour_group_name"], transactions["index_name"]))
        
        """negative sampling"""
        # set up negative:positive ratio as 4:1
        negative_samples = 4

        for u, i, price, sale, club, age, product, color, index in tqdm(customer_product_set):
            customers.append(u)
            products.append(i)
            prices.append(price)
            sales_channels.append(sale)
            club_status.append(club)
            age_groups.append(age)
            product_groups.append(product)
            color_groups.append(color)
            index_name.append(index)
            labels.append(1)
            for _ in range(negative_samples):
                negative_product = np.random.choice(all_products_id)
                while (u, negative_product, price, sale, club, age, product, color, index) in customer_product_set:
                    negative_product = np.random.choice(all_products_id)
                customers.append(u)
                products.append(negative_product)
                prices.append(price)
                sales_channels.append(sale)
                club_status.append(club)
                age_groups.append(age)
                product_groups.append(product)
                color_groups.append(color)
                index_name.append(index)
                labels.append(0)
        return customers, products, prices, sales_channels, club_status, age_groups, product_groups, color_groups, index_name, torch.tensor(labels)
    
def binary_acc(y_pred, y_test):
    acc = 0.0
    
    y_pred_label = torch.softmax(y_pred, dim=1)
    _, y_pred_label = torch.max(y_pred_label, dim = 1)
    correct_pred = (y_pred_label == y_test).sum()
    print(y_test.shape[0])
    acc = correct_pred.item()/y_test.shape[0]
    return acc
     
    


def sparse_to_tensor(sparse_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    sparse_matrix = sparse_matrix.tocoo()
    values = sparse_matrix.data
    indices = (sparse_matrix.row, sparse_matrix.col) # np.vstack
    shape = sparse_matrix.shape

    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    s = torch.Size(shape)

    return torch.sparse.DoubleTensor(i, v, s)

def sparse_batch_collate(batch): 
    """
    Collate function which to transform scipy csr matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list

    customer_batch, product_batch, targets_batch = batch[0]
    
    if type(customer_batch[0]) == csr_matrix:
        customer_batch = customer_batch.tocoo() # removed vstack
        customer_batch = sparse_to_tensor(customer_batch)
    else:
        customer_batch = torch.DoubleTensor(customer_batch)

    if type(product_batch[0]) == csr_matrix:
        product_batch = product_batch.tocoo() # removed vstack
        product_batch = sparse_to_tensor(product_batch)
    else:
        product_batch = torch.DoubleTensor(product_batch)
    
    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo() # removed vstack
        targets_batch = sparse_to_tensor(targets_batch)
    else:
        targets_batch = torch.DoubleTensor(targets_batch)
    
    return customer_batch, product_batch, targets_batch