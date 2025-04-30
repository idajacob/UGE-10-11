########## IMPORTS ##########
import json
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes
from torch import nn
from torchvision import datasets, transforms
from network import neural_network


########## FORBINDELSE TIL CONFIG ##########

# forbindelse til config.json
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

config = load_config('src/config.json')
epochs = config['epochs']
learning_rate = config['learning_rate']
batch_size = config['batch_size']


########## OPSÆTNING AF DEVICE ##########

# opsætning af device - tjekker om GPU findes og ellers kører den på CPU. GPU er bedre (ligesom med MAX)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


########## LOAD DATA ##########

# billeder blver til tensor
transform = transforms.ToTensor()

# trænings- og testdata
train_dataset = datasets.FashionMNIST(
    root="data",
    train=True,
    download=False,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=transform
)

# dataloader
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


########## OPRET MODEL, OPTIMIZER OG LOSS ##########

# opret model - flyt modellen til enten GPU eller CPU alt efter tilgængelighed
model = neural_network().to(device)

optimizer = torch.optim.Adam(params = model.parameters(), lr = learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


########## TEST OG TRAIN LOOPS ##########

model_file = "outputs/model_weights.pt"
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.eval()

# Train funktion
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()  # Sæt model til træningstilstand
    total_loss = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        # Fremadpropagering
        pred = model(X)
        loss = loss_fn(pred, y)

        # Tilbagepropagering
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


# Test funktion
def test(dataloader, model, loss_fn, device):
    model.eval()  # Sæt model til evaluerings-tilstand
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy

torch.save(model.state_dict(), "outputs/model_weights.pt")


########## LISTER, GRAFER OG BILLEDER ##########

# lister til resultater
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    test_loss, test_accuracy = test(test_dataloader, model, loss_fn, device)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    # print i terminal så jeg ved hvad der sker
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")
    
    # gem i en fil
    with open("outputs/training_log.txt", "a") as f:
        f.write(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}\n")

# tegn grafer
epochs_range = range(1, epochs + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loos over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_plot.png")
plt.show()


# tegn images
def show_more_image(dataset, num_image=24):
    cols = rows = int(num_image ** 0.5)
    
    axes: np.ndarray
    
    figur,axes = plt.subplots(rows,cols)
    
    for ax, (image, label_number) in zip(axes.flat, dataset):
        image:torch.Tensor
        
        ax.imshow(image.squeeze(),cmap="grey")
        ax.set_title(f"Label: {label_number}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("outputs/test_images.png")
    plt.show()

show_more_image(test_dataset)