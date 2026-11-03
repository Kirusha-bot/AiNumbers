import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.1

device = torch.device("cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class DigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.hidden = nn.Linear(28 * 28, 128)

        self.activation = nn.ReLU()

        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

model = DigitNet().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train_one_epoch():
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total

if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        avg_loss = train_one_epoch()
        accuracy = evaluate()
        print(f"Эпоха {epoch}/{EPOCHS} | ошибка: {avg_loss:.4f} | точность на тесте: {accuracy * 100:.2f}%")

    torch.save(model.state_dict(), "digit_net.pth")
    print("Модель сохранена в digit_net.pth")


    model.eval()
    for i in range(10):
        image, label = test_dataset[i]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            probs = torch.softmax(output, dim=1)
            pred = probs.argmax(dim=1).item()
        print(f"Правильная цифра: {label}, сеть думает: {pred} (уверенность {probs.max().item() * 100:.1f}%)")
