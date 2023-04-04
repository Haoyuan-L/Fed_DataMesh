import torch
from torch.utils.data import Dataset


class CreditCardDataLoader(Dataset):
    """CreditCardDataLoader dataloader for credit card dataset

    Args:
        digitial_transaction (pd.DataFrame): Dataframe of digital transaction records
        retail_transaction (pd.DataFrame): Dataframe of retail transaction records
        fraud_prevention (pd.DataFrame): Dataframe of fraud prevention records
        label (pd.Series): Series of labels
    """
    def __init__(self, digital_transaction, retail_transaction, fraud_prevention, label):
        self.digital_transaction = digital_transaction
        self.retail_transaction = retail_transaction
        self.fraud_prevention = fraud_prevention
        self.label = label
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.digital_transaction[idx], dtype=torch.float32),
            torch.tensor(self.retail_transaction[idx], dtype=torch.float32),
            torch.tensor(self.fraud_prevention[idx], dtype=torch.float32),
            torch.tensor(self.label[idx], dtype=torch.long)
        )

# distribute credit card data for each data owner of domain
class Distributed_CreditCard:
    def __init__(self, data_owners, data_loader):
        self.data_owners = data_owners
        self.data_loader = data_loader
        self.no_of_owner = len(data_owners)

        # set up list for data pointers
        self.data_pointer = []
        self.labels = []

        # iterate over each batch of dataloader, split data based on domains, sending to VirtualWorker 
        for digital_transaction_batch, retail_transaction_batch, fraud_prevention_batch, label_batch in data_loader:
            curr_data_dict = {}
            self.labels.append(label_batch)
            digital_part_ptr = digital_transaction_batch.send(self.data_owners[0])
            retail_part_ptr = retail_transaction_batch.send(self.data_owners[1])
            fraud_part_ptr = fraud_prevention_batch.send(self.data_owners[2])
            data_part_ptr = [digital_part_ptr, retail_part_ptr, fraud_part_ptr]
            # send data to data owner
            for i, data_owner in enumerate(self.data_owners):         
                curr_data_dict[data_owner.id] = data_part_ptr[i]
            # add data pointer to the list
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