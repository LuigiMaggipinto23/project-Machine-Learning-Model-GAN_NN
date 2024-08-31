import torch.utils.data as dutils
import os
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from constants import IMG_SIZE, BATCH_SIZE

dir_train = os.path.join('dataset', 'train')
dir_test =  os.path.join('dataset', 'test')
dir_val =   os.path.join('dataset', 'val')

class CustomDataset(dutils.Dataset):
    def __init__(self, transform=None, dir=dir_train):
        self.dir = dir
        self.image_names = os.listdir(dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = read_image(os.path.join(self.dir, image_name)).float() / 255.0  # Normalizzare l'immagine
        if self.transform:
            image = self.transform(image)
        return image

transform = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(10),
    T.CenterCrop(192),
    T.RandomCrop(180),
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

dataset_train = CustomDataset(transform=transform, dir=dir_train)
dataset_test = CustomDataset(transform=transform, dir=dir_test)
dataset_val = CustomDataset(transform=transform, dir=dir_val)

dataloader_train = dutils.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_test  = dutils.DataLoader(dataset_test,  batch_size=BATCH_SIZE, shuffle=True)
dataloader_val   = dutils.DataLoader(dataset_val,   batch_size=BATCH_SIZE, shuffle=True)


if __name__ == '__main__':
    images = next(iter(dataloader_train))
    plt.figure(figsize=(8, 8))
    plt.title('Images')
    plt.axis('off')
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.imshow(T.ToPILImage()(images[i]))
    plt.show()