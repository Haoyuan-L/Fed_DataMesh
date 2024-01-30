# Empowering data mesh with federated learning


## Introduction
The evolution of data architecture has seen the rise of data lakes, aiming to solve the bottlenecks of data management and promote intelligent decision-making. However, this centralized architecture is limited by the proliferation of data sources and the growing demand for timely analysis and processing. A new data paradigm, Data Mesh, is proposed to overcome these challenges. Data Mesh treats domains as a first-class concern by distributing the data ownership from the central team to each data domain, while keeping the federated governance to monitor domains and their data products. Many multi-million dollar organizations like Paypal, Netflix, and Zalando have already transformed their data analysis pipelines based on this new architecture.\\
In this decentralized architecture where data is locally preserved by each domain team, traditional centralized machine learning is incapable of conducting effective analysis across multiple domains, especially for security-sensitive organizations. To this end, we introduce a pioneering approach that incorporates Federated Learning into Data Mesh.

## Environment Setup:
Install Conda Environment via enviroment.yml

``` conda env create -f enviroment.yml ```

## Centralized Training
- Fraud Detection: Centralized_FraudNN
- Recommendation System: Centralized_RecNN

## Federated Training
- Fraud Detection: Split_FraudNN
- Recommendation System: Split_RecNN

## Contributors
- Haoyuan Li
- Salman Toor
