import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataloader import RyersonEmotionDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np


#  Hyperparameters 
learning_rate = 0.001
num_epochs = 10 
batch_size = 32  


#  CNNMLP Architecture 
class CNN_MLP(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=4, stride=4), 
            nn.Dropout(0.25),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(0.25),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128), 
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)
        return x


#  Initialize 
model = CNN_MLP(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate) # RMSprop


#  DataLoader 
dataset = RyersonEmotionDataset('processed')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#  Training Loop 
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
            .format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))

print("Finished Training")

#  Save the Model 
torch.save(model.state_dict(), 'emotion_model.pth')
print("Model saved to emotion_model.pth")

# Evaluation with Metrics 
all_preds = []
all_labels = []

total_correct = 0
total_samples = 0
with torch.no_grad(): # Disable gradient calculation for evaluation
    for i, (images, labels) in enumerate(test_loader):
        print(i)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples * 100
    print(f'Accuracy of the model on the test images: {accuracy:.4f} %')

# Calculate metrics
precision = precision_score(all_labels, all_preds, average=None)
recall = recall_score(all_labels, all_preds, average=None)
f1 = f1_score(all_labels, all_preds, average=None)

# Emotion labels 
emotion_labels = ['Happy', 'Sad', 'Angry', 'Fear', 'Surprise', 'Disgust']

# Print the metrics in a table format
print("\nEvaluation Metrics:")
print("{:<10} {:<10} {:<10} {:<10}".format("Emotion", "Precision", "Recall", "F1-score"))
for i, label in enumerate(emotion_labels):
    print("{:<10} {:<10.2f} {:<10.2f} {:<10.2f}".format(label, precision[i] * 100, recall[i] * 100, f1[i] * 100))
