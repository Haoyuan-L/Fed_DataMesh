import torch
import torch.nn as nn

class NCF(nn.Module):
    """NCF - Neural Collaborative Filtering proposed by He et al.

    Args:
        num_users (int): Number of users
        num_items (iut): Number of products
        transactions (pd.DataFrame): Dataframe of transaction records
        all_products_id (list): A list contains all product ids
    """
    def __init__(
            self, 
            num_users, 
            num_items, 
            embedding_dim: int = 10,
            hidden_size: int = 64,
            output_size: int = 32,
            num_hidden_layers: int = 1,
            num_classes: int = 2,
        ):
            super().__init__()
            self.user_embedding_layer = nn.Embedding(num_embeddings=num_users, embedding_dim=embedding_dim)
            self.item_embedding_layer = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
            self.relu = nn.ReLU()
            self.fcs = nn.Sequential()
            if embedding_dim is not None:
                in_channels = (
                    [embedding_dim + embedding_dim] 
                    + [hidden_size]*num_hidden_layers 
                    + [output_size]
                )
            else:
                raise ValueError
            for i in range(len(in_channels)):
                if i != len(in_channels)-1:
                    self.fcs.append(nn.Linear(in_features=in_channels[i],  out_features=in_channels[i+1]))
                else:
                    self.fcs.append(nn.Linear(in_features=in_channels[i],  out_features=num_classes))
    
    def forward(self, user_input, item_input):

        user_embedding = self.user_embedding_layer(user_input)
        item_embedding = self.item_embedding_layer(item_input)
        concat_embedding = torch.cat([user_embedding, item_embedding], dim=-1)
        
        for fc_layer in self.fcs:
            concat_embedding = fc_layer(concat_embedding)
            concat_embedding = self.relu(concat_embedding)
        # print("TEST")
        # print(concat_embedding)
        #pred = F.softmax(concat_embedding)
        return concat_embedding
    