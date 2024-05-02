#Importazione del modulo json --> Per l'implementazione del file .json
import json

from nltk_utils import tokenize, stem, bag_of_words
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model import NeuralNet

# Legge il contenuto di un file JSON chiamato intents.json e carica un dizionario chiamato intents. i dati letti sul file JSON vengono inizializzati su una variabile intents
# 'r' sta per read mode, ossia modalità di lettura
with open('intents.json','r') as f:
    intents = json.load(f)

all_words = [] #lista vuota
tags = [] #lista vuota per i tags
xy = [] #lista vuota per i pattern e i tags

# Ciclo for che itera per ogni intent nella lista intents nella "categoria" intents del JSON
for intent in intents['intents']:
    
    # Nella variabile tag viene inserito il 'tag' di intent
    tag = intent['tag']
    
    # Aggiunge il tag alla lista dei tags
    tags.append(tag)
    
    # Ciclo for che itera per ogni pattern nella lista patterns di intent
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        
        # Aggiunge tutte le parole alla lista all_words
        all_words.extend(w)
        
        # Aggiunge il pattern e il tag alla lista xy
        xy.append((w,tag))

ignore_words = ['?','!','.',',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))
#print("i tags sono:")
#print(tags)
#print(xy)
#print(w)
#print(all_words)

X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence,all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label) # CrossEntropoyLoss

#print(X_train)   

X_train = np.array(X_train)
y_train = np.array(y_train)

num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

class ChatDataset(Dataset):
     def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
     def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
     def __len__(self):
        return self.n_samples


#print(input_size, len(all_words))
#print(output_size, tags)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size,hidden_size,output_size).to(device)

#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')