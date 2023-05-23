import numpy as np
import torch
import torchvision.models as models
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from utils.DogBreedDataset import DogBreedDataset


def train(net, label_dict, learning_rate=1e-4, num_epochs=100):
    train_dataset = DogBreedDataset(split="train", label_dict=label_dict)
    val_dataset = DogBreedDataset(split="val", label_dict=label_dict)

    training_generator = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_generator = DataLoader(val_dataset, batch_size=32, shuffle=True)

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for i in range(num_epochs):
        net.train()  # nouveau
        for j, sample in enumerate(training_generator):
            x, y = sample
            y = y.to(torch.long).cuda()
            optimizer.zero_grad()
            out = net(x.cuda())
            loss = loss_func(out, y.cuda())
            loss.backward()
            optimizer.step()
            print('\r Epoch', i, 'Step', j, ':', str(loss.data.cpu().numpy()), end="")

        net.eval()  # nouveau
        accuracy = []
        for j, sample in enumerate(test_generator):
            x, y = sample
            y = y.to(torch.long).cuda()
            out = net(x.cuda())
            best = np.argmax(out.data.cpu().numpy(), axis=-1)
            accuracy.extend(list(best == y.data.cpu().numpy()))
        print('\n Accuracy is ', str(np.mean(accuracy) * 100))
    print("done")


if __name__ == "__main__":
    label_dict = dict(
        n02086240='Shih-Tzu',
        n02087394='Rhodesian ridgeback',
        n02088364='Beagle',
        n02089973='English foxhound',
        n02093754='Australian terrier',
        n02096294='Border terrier',
        n02099601='Golden retriever',
        n02105641='Old English sheepdog',
        n02111889='Samoyed',
        n02115641='Dingo'
    )
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features=in_features, out_features=len(label_dict.keys()))
    model = model.cuda()
    train(model, label_dict, num_epochs=3)
    torch.save(model.state_dict(), "models/resnet_pretrained_dog_breed.pt")
