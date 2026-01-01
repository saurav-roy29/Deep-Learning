import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# --- 1. CONFIGURATION & DATA PREP ---
# We normalize images so pixel values are between -1 and 1 (helps training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print("Downloading dataset... (this might take a minute)")
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# --- 2. THE CNN ARCHITECTURE ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer 1: Convolution
        # Input: 3 channels (RGB), Output: 6 features, Kernel: 5x5
        self.conv1 = nn.Conv2d(3, 6, 5) 
        self.pool = nn.MaxPool2d(2, 2)  # Shrink by half (2x2 grid)
        
        # Layer 2: Convolution
        # Input: 6 features (from prev), Output: 16 features, Kernel: 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # Layer 3: Fully Connected (Standard Neural Net)
        # We need to flatten the 3D image features into a 1D vector.
        # Math: 16 channels * 5 * 5 image size = 400 inputs
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10 Outputs (for the 10 classes)

        self.relu = nn.ReLU() # Our trusty activation function

    def forward(self, x):
        # Pass through Conv1 -> ReLU -> Pool
        x = self.pool(self.relu(self.conv1(x)))
        
        # Pass through Conv2 -> ReLU -> Pool
        x = self.pool(self.relu(self.conv2(x)))
        
        # FLATTEN: Turn the 3D tensor (16x5x5) into a 1D vector (400)
        x = torch.flatten(x, 1) 
        
        # Pass through Feedforward layers
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # No activation here! CrossEntropyLoss handles the Softmax.
        return x

model = SimpleCNN()

# --- 3. LOSS & OPTIMIZER ---
criterion = nn.CrossEntropyLoss() # Standard for Multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# --- 4. TRAINING LOOP ---
print("\n--- Start Training ---")
for epoch in range(2):  # Loop over dataset 2 times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        # A. Zero the gradients
        optimizer.zero_grad()

        # B. Forward + Backward + Optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # Print every 2000 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

# --- 5. TEST ON ONE IMAGE ---
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Print the real label
print(f"\nReal Label: {classes[labels[0]]}")

# Ask the model
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print(f"Model Prediction: {classes[predicted[0]]}")