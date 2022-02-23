from data import PetDataset
import torch
from torchvision import models
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import pandas as pd


if __name__ == '__main__':

    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_data = PetDataset(annotation_file='data/myannotations_test.csv', split='test')


    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, test_data.num_classes)
    model.load_state_dict(torch.load('files/best_model.pth'))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    # evaluate
    since = time.time()

    model.eval()
    running_loss = 0.0
    running_corrects = 0.0

    with torch.no_grad():
        for e, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if e % 5 == 0:
                print('Phase: Test, Batch: {}/{}, Loss: {}'.format(
                    e + 1, len(test_data) // batch_size, loss.item()
                ))

        total_loss = running_loss / len(test_data)
        total_acc = running_corrects.double() / len(test_data)

    print('Test Loss: {}, Test Accuracy: {}'.format(total_loss, total_acc))

