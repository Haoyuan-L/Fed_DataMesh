import torch
import torch.nn as nn

class NCF(nn.Module):
    """NCF - Neural Collaborative Filtering proposed by He et al.

    Args:
        num_users (int): Number of users
        num_items (iut): Number of products
        num_product_groups (int): Number of product groups
        num_color_groups: (int): Number of color groups
        num_index_name: (int): Number of index name
    """
    def __init__(
            self, 
            num_users: int, 
            num_items: int,
            num_product_groups: int,
            num_color_groups: int,
            num_index_name: int,
            user_embedding_dim: int = 16,
            item_embedding_dim: int = 32,
            product_group_embedding_dim: int = 10,
            color_group_embedding_dim: int = 8,
            index_name_embedding_dim: int = 6,
            input_size: int = 76,
            hidden_size_1: int = 64,
            hidden_size_2: int = 128,
            output_size: int = 32,
            num_hidden_layers: int = 2,
            num_classes: int = 2,
        ):
            super().__init__()
            # embedding layers for categorical features and user-item interaction
            self.user_embedding_layer = nn.Embedding(num_embeddings=num_users, embedding_dim=user_embedding_dim)
            self.item_embedding_layer = nn.Embedding(num_embeddings=num_items, embedding_dim=item_embedding_dim)
            self.product_group_embedding_layer = nn.Embedding(num_embeddings=num_product_groups, embedding_dim=product_group_embedding_dim)
            self.color_group_embedding_layer = nn.Embedding(num_embeddings=num_color_groups, embedding_dim=color_group_embedding_dim)
            self.index_name_embedding_layer = nn.Embedding(num_embeddings=num_index_name, embedding_dim=index_name_embedding_dim)

            self.relu = nn.LeakyReLU()
            self.fcs = nn.Sequential()
            if user_embedding_dim and item_embedding_dim is not None:
                in_channels = (
                    [input_size] 
                    + [hidden_size_2 + hidden_size_2]*num_hidden_layers 
                    + [hidden_size_2]
                    + [hidden_size_1]
                    + [output_size]
                )
            else:
                raise ValueError
            for i in range(len(in_channels)):
                if i != len(in_channels)-1:
                    self.fcs.append(nn.Linear(in_features=in_channels[i],  out_features=in_channels[i+1]))
                else:
                    self.fcs.append(nn.Linear(in_features=in_channels[i],  out_features=num_classes))
    
    def forward(self, user_input, item_input, prices, sales_channels, club_status, age_groups, product_groups, color_groups, index_name):

        user_embedding = self.user_embedding_layer(user_input)
        item_embedding = self.item_embedding_layer(item_input)
        product_group_embedding = self.product_group_embedding_layer(product_groups)
        color_group_embedding = self.color_group_embedding_layer(color_groups)
        index_name_embedding = self.index_name_embedding_layer(index_name)
        concat_embedding = torch.cat([user_embedding, item_embedding, product_group_embedding, color_group_embedding, index_name_embedding], dim=-1)

        concat_input = torch.cat([prices.reshape(-1, 1), sales_channels.reshape(-1, 1), club_status.reshape(-1, 1), age_groups.reshape(-1, 1)], dim=-1)
        concat_embedding = torch.cat([concat_embedding, concat_input], dim=-1)

        for fc_layer in self.fcs:
            concat_embedding = fc_layer(concat_embedding)
            concat_embedding = self.relu(concat_embedding)
        #pred = F.softmax(concat_embedding)
        return concat_embedding
    