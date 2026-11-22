import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# 1. Define image transformations (Resize and normalise for ResNet)
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 2. Load datasets (expects a 'dataset' folder with 'train' and 'val' subfolders)
data_dir = 'dataset'
if not os.path.exists(data_dir):
    print(f"Warning: '{data_dir}' directory not found. Please create it and add your image classes.")
else:
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                      for x in ['train', 'val']}

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=2) 
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"Classes found: {class_names}")

    # Set device to GPU if available, else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 3. Load Pre-trained ResNet18 and modify the final layer for your classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze early layers so I'm only train the final classification head
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer to match your number of classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. Define Loss Function and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 5. Training Loop
    num_epochs = 5

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = torch.tensor(0.0, device=device)

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimise only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    print("Training complete!")

    # Save your trained model weights
    torch.save(model.state_dict(), 'astronomy_model.pth')
    print("Model saved as astronomy_model.pth")
