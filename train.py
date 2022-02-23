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
    num_epochs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    train_data = PetDataset(annotation_file='data/myannotations_train.csv')
    val_data = PetDataset(annotation_file='data/myannotations_val.csv', split='val')

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, train_data.num_classes)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # train
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        running_corrects = 0.0

        for e, train_batch in enumerate(train_loader):
            images = train_batch['image'].to(device)
            labels = train_batch['label'].to(device)

            optimizer.zero_grad()

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss  += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)

            if e % 5 == 0:
                print('Phase: Train, Epoch: {}/{}, Batch: {}/{}, Loss: {}'.format(
                    epoch+1, num_epochs, e+1, len(train_data) // batch_size, loss.item()
                ))

        exp_lr_scheduler.step()

        epoch_loss = running_loss / len(train_data)
        epoch_acc = running_corrects.double() / len(train_data)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0.0

        with torch.no_grad():
            for e, val_batch in enumerate(val_loader):
                images = val_batch['image'].to(device)
                labels = val_batch['label'].to(device)

                outputs = model(images)

                _, preds = torch.max(outputs, 1)

                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

                if e % 5 == 0:
                    print('Phase: Val, Epoch: {}/{}, Batch: {}/{}, Loss: {}'.format(
                        epoch+1, num_epochs, e + 1, len(val_data) // batch_size, loss.item()
                    ))

            val_epoch_loss = val_running_loss / len(val_data)
            val_epoch_acc = val_running_corrects.double() / len(val_data)

            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print('Epoch: {}/{}, train_loss: {}, train_acc: {}, val_loss: {}, val_acc: {}, best_acc: {}'.format(
            epoch+1, num_epochs, epoch_loss, epoch_acc, val_epoch_loss, val_epoch_acc, best_acc
        ))
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.cpu().numpy())
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc.cpu().numpy())
        print('types: {} {} {} {}'.format(type(epoch_loss), type(epoch_acc), type(val_epoch_loss), type(val_epoch_acc)))

    df = pd.DataFrame(history)
    df.to_csv('files/history.csv', index=False)
    torch.save(best_model_wts, 'files/best_model.pth')































