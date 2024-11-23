import torch

# Loss function and optimizer
criterion = torch.nn.MSELoss()

# Carregar os batches salvos no pré-processamento
train_batches = torch.load("data/train_batches.pt")
test_batches = torch.load("data/test_batches.pt")