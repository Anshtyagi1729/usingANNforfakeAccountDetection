import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from torch.utils.data import DataLoader,TensorDataset

#loading the dataset
insta_df_train=pd.read_csv('insta_train.csv')
insta_df_test=pd.read_csv('insta_test.csv')
#display first few rows
print(insta_df_train.head())
print(insta_df_test.head())
#eda
# Perform EDA
insta_df_train.info()
insta_df_train.describe()
print(insta_df_train.isnull().sum())

# Visualization
sns.countplot(insta_df_train['fake'])
plt.show()

sns.countplot(insta_df_train['private'])
plt.show()

sns.countplot(insta_df_train['profile pic'])
plt.show()

plt.figure(figsize=(20, 10))
sns.histplot(insta_df_train['nums/length username'])
plt.show()

plt.figure(figsize=(20, 20))
sns.heatmap(insta_df_train.corr(), annot=True)
plt.show()
#preparing data for the torch model
# Preparing data
X_train = insta_df_train.drop(columns=['fake'])
X_test = insta_df_test.drop(columns=['fake'])
y_train = insta_df_train['fake']
y_test = insta_df_test['fake']

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#defining the model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(11, 50)
        self.fc2 = nn.Linear(50, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 25)
        self.fc5 = nn.Linear(25, 25)
        self.fc6 = nn.Linear(25, 2)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x
#training 
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

train_model(model, train_loader, criterion, optimizer)

#eval
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds

y_test_actual, y_test_pred = evaluate_model(model, test_loader)

print(classification_report(y_test_actual, y_test_pred))

cm = confusion_matrix(y_test_actual, y_test_pred)
plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True)
plt.show()

