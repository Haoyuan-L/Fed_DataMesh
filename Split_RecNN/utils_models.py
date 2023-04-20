import torch
from torch import nn

class SalesNN(nn.Module):
    """ Partial model for sales domain
    Args:
        num_users (int): Number of users
        num_items (int): Number of products
        prices (float): price of transactions
        sales_channels (float): sales channels
    
    """
    def __init__(
            self, 
            num_users: int, 
            num_items: int,
            input_size: int = 48,
            user_embedding_dim: int = 32,
            item_embedding_dim: int = 64,
            hidden_size_1: int = 128,
            output_size: int = 32,
        ):
        super().__init__()
        self.user_embedding_layer = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embedding_dim)
        self.item_embedding_layer = nn.Embedding(num_embeddings=num_items, embedding_dim=item_embedding_dim)
        
    def forward(self, user_input, item_input):
        user_embedding = self.user_embedding_layer(user_input)
        item_embedding = self.item_embedding_layer(item_input)

        latent_vec = torch.cat([user_embedding, item_embedding], dim=-1)

        return latent_vec
    
    # save weights of partial model on remote worker
    def get_weights(self):
        return self.state_dict()
    
class CustomersNN(nn.Module):
    """ Partial model for customer domain
    Args:
        club_status (int): active or inactive customers' status
        age_groups (int): age of customers
    
    """
    def __init__(
            self,
            input_size: int = 2,
            output_size: int = 5,
        ):
        super().__init__()
        self.relu = nn.LeakyReLU()
        in_channels = (
            [input_size] 
            + [output_size]
        )
        
        self.encoder = nn.Sequential(
            *[nn.Linear(in_features=in_channels[i], out_features=in_channels[i+1]) for i in range(len(in_channels)-1) if i != len(in_channels)-1]
        )
        
    def forward(self, club_status, age_groups):
        
        latent_vec = torch.cat([club_status, age_groups], dim=-1)
        
        for layer in self.encoder:
            latent_vec = layer(latent_vec)
        
        return latent_vec
    
    # save weights of partial model on remote worker
    def get_weights(self):
        return self.state_dict()

class ProductsNN(nn.Module):
    """ Partial model for product domain
    Args:
        num_product_groups (int): Number of product groups
        num_color_groups: (int): Number of color groups
        num_index_name: (int): Number of index name
    
    """
    def __init__(
            self,
            num_product_groups: int,
            num_color_groups: int,
            num_index_name: int,
            product_group_embedding_dim: int = 8,
            color_group_embedding_dim: int = 16,
            index_name_embedding_dim: int = 6,
        ):
        super().__init__()
        self.product_group_embedding_layer = nn.Embedding(num_embeddings=num_product_groups, embedding_dim=product_group_embedding_dim)
        self.color_group_embedding_layer = nn.Embedding(num_embeddings=num_color_groups, embedding_dim=color_group_embedding_dim)
        self.index_name_embedding_layer = nn.Embedding(num_embeddings=num_index_name, embedding_dim=index_name_embedding_dim)
        

        
    def forward(self, product_groups, color_groups, index_name):
        product_group_embedding = self.product_group_embedding_layer(product_groups)
        color_group_embedding = self.color_group_embedding_layer(color_groups)
        index_name_embedding = self.index_name_embedding_layer(index_name)

        latent_vec = torch.cat([product_group_embedding, color_group_embedding, index_name_embedding], dim=-1)
        
        return latent_vec
    
    # save weights of partial model on remote worker
    def get_weights(self):
        return self.state_dict()