import torch

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