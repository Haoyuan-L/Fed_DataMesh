{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class Distributed_HM:\n",
    "    def __init__(self, data_owners, data_loader):\n",
    "        self.data_owners = data_owners\n",
    "        self.data_loader = data_loader\n",
    "        self.no_of_owner = len(data_owners)\n",
    "\n",
    "        self.data_pointer = []\n",
    "        self.labels = []\n",
    "\n",
    "        # iterate over each batch of dataloader, split data based on domains, sending to VirtualWorker  \n",
    "        for customer_batch, product_batch, prices_batch, sales_channels_batch, club_status_batch, age_groups_batch, product_groups_batch, color_groups_batch, index_name_batch, label_batch in data_loader:\n",
    "            \n",
    "            curr_data_dict = {}\n",
    "            self.labels.append(label_batch)\n",
    "\n",
    "            # split data batch based on domains\n",
    "            sales_domain = torch.cat([customer_batch.reshape(-1, 1), product_batch.reshape(-1, 1), sales_channels_batch.reshape(-1, 1), prices_batch.reshape(-1, 1)], dim=1)\n",
    "            customer_domain = torch.cat([club_status_batch.reshape(-1, 1), age_groups_batch.reshape(-1, 1)])\n",
    "            product_domain = torch.cat([product_groups_batch.reshape(-1, 1), color_groups_batch.reshape(-1, 1), index_name_batch.reshape(-1, 1)])\n",
    "\n",
    "            # set data owners for each domain team\n",
    "            sales_owner = self.data_owners[0]\n",
    "            customer_owner = self.data_owners[1]\n",
    "            product_owner = self.data_owners[2]\n",
    "\n",
    "            # send split data to VirtualWorkers and add the data pointer to the dict\n",
    "            sales_part_ptr = sales_domain.send(sales_owner)\n",
    "            curr_data_dict[sales_owner.id] = sales_part_ptr\n",
    "            customer_part_ptr = customer_domain.send(customer_owner)\n",
    "            curr_data_dict[customer_owner.id] = customer_part_ptr\n",
    "            product_part_ptr = product_domain.send(product_owner)\n",
    "            curr_data_dict[product_owner.id] = product_part_ptr\n",
    "\n",
    "            self.data_pointer.append(curr_data_dict)\n",
    "\n",
    "    def __iter__(self):\n",
    "        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):\n",
    "            yield (data_ptr, label)\n",
    "    \n",
    "    def __len__(self):\n",
    "        return len(self.data_loader)-1\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
