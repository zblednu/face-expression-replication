import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataloader import RyersonEmotionDataset


# --- Hyperparameters (Paper-Specific) ---
learning_rate = 0.001
num_epochs = 10  # You might need to adjust this based on your experiments
batch_size = 32  # Or whatever batch size you can handle


# --- Define CNN-MLP Architecture (Paper-Specific) ---
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


# --- Initialize ---
model = CNN_MLP(num_classes=6)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate) # RMSprop


# --- DataLoader ---
dataset = RyersonEmotionDataset('processed')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# --- Training Loop ---
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


# --- Evaluation ---
model.eval() # Set the model to evaluation mode
total_correct = 0
total_samples = 0
with torch.no_grad(): # Disable gradient calculation for evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    print(f'Accuracy of the model on the test images: {accuracy:.4f} %')


# --- Save the Model ---
torch.save(model.state_dict(), 'emotion_model.pth')
print("Model saved to emotion_model.pth")
