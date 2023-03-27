import torch
from torch.utils.data import Dataset
from tqdm.notebook import tqdm
import numpy as np

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
        
        customers = torch.tensor(customers)
        products = torch.tensor(products)
        prices = torch.tensor(prices)
        sales_channels = torch.tensor(sales_channels)
        club_status = torch.tensor(club_status)
        age_groups = torch.tensor(age_groups)
        product_groups = torch.tensor(product_groups)
        color_groups = torch.tensor(color_groups)
        index_name = torch.tensor(index_name)
        labels = torch.tensor(labels)
        
        return customers, products, prices, sales_channels, club_status, age_groups, product_groups, color_groups, index_name, labels


class Distributed_HM:
    def __init__(self, data_owners, data_loader):
        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        self.data_pointer = []
        self.labels = []
#         self.device = device

        # iterate over each batch of dataloader, split data based on domains, sending to VirtualWorker  
        for customer_batch, product_batch, prices_batch, sales_channels_batch, club_status_batch, age_groups_batch, product_groups_batch, color_groups_batch, index_name_batch, label_batch in data_loader:
            
            curr_data_dict = {}
            self.labels.append(label_batch)

            # split data batch based on domains
            sales_domain = [customer_batch, product_batch, sales_channels_batch.float().reshape(-1, 1), prices_batch.reshape(-1, 1)]
            customer_domain = [club_status_batch.float().reshape(-1, 1), age_groups_batch.reshape(-1, 1)]
            product_domain = [product_groups_batch, color_groups_batch, index_name_batch]
            
#             # Move tensors to the GPU
#             sales_domain = [tensor.to(self.device) for tensor in sales_domain]
#             customer_domain = [tensor.to(self.device) for tensor in customer_domain]
#             product_domain = [tensor.to(self.device) for tensor in product_domain]
            
            # set data owners for each domain team
            sales_owner = self.data_owners[0]
            customer_owner = self.data_owners[1]
            product_owner = self.data_owners[2]
            
            
            # send split data to VirtualWorkers and add the data pointer to the dict
            sales_part_ptr = []
            for tensor in sales_domain:
                sales_part_ptr.append(tensor.send(sales_owner))
            curr_data_dict[sales_owner.id] = sales_part_ptr
            
            customer_part_ptr = []
            for tensor in customer_domain:
                customer_part_ptr.append(tensor.send(customer_owner))
            curr_data_dict[customer_owner.id] = customer_part_ptr
            
            product_part_ptr = []
            for tensor in product_domain:
                product_part_ptr.append(tensor.send(product_owner))
            curr_data_dict[product_owner.id] = product_part_ptr

            self.data_pointer.append(curr_data_dict)

    def __iter__(self):
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield (data_ptr, label)
    
    def __len__(self):
        return len(self.data_loader)-1

def binary_acc(y_pred, y_test):
    acc = 0.0
    y_pred_label = torch.softmax(y_pred, dim=1)
    _, y_pred_label = torch.max(y_pred_label, dim = 1)
    correct_pred = (y_pred_label == y_test).sum()
    acc = correct_pred.item()/y_test.shape[0]
    
    return acc 