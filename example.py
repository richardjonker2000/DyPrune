#  Should you have multiple cuda devices and would like to select only one
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from DyPrune import DyPrune
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torchsummary import summary
from sklearn import preprocessing
import pandas as pd
import numpy as np


#download and load the fashionmnist dataset
def load_data_fashion_mnist():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )
  #split the dataset
    training_data, validation_data = torch.utils.data.random_split(training_data, [50000, 10000],
                                                                   generator=torch.Generator().manual_seed(42))

    # create dataloaders, default 128 batch size
    train_dataloader = DataLoader(training_data, batch_size=128, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=128, num_workers=8, pin_memory=True)
    validation_dataloader = DataLoader(validation_data, batch_size=128, num_workers=8, pin_memory=True)

    return train_dataloader, test_dataloader, validation_dataloader

#download the iris dataset
class IrisDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file containing the iris.data dataset https://archive.ics.uci.edu/ml/datasets/iris.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.iris = pd.read_csv(csv_file, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

        self.classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

        self.labels = self.iris.pop("class")
        values = self.iris.values
        self.iris = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(values))
        self.transform = transform

    def __len__(self):
        return len(self.iris)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # data = self.iris.iloc[idx]
        target = self.classes.index(self.labels.iloc[idx])
        data = torch.tensor(self.iris.iloc[idx].to_list(), dtype=torch.float32)
        if self.transform:
            sample = self.transform(data)

        return data, target

#load the iris dataset, same implemenatation as the fashionmnist dataset
def load_data_iris():
    data = IrisDataset("iris/iris.data")

    training_data, validation_data, test_data = torch.utils.data.random_split(data, [100, 25, 25],
                                                                              generator=torch.Generator().manual_seed(
                                                                                  42))

    # create dataloaders, default 128
    train_dataloader = DataLoader(training_data, batch_size=12, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=12, num_workers=8)
    validation_dataloader = DataLoader(validation_data, batch_size=12, num_workers=8)

    return train_dataloader, test_dataloader, validation_dataloader

#a simple main function to load onne of the models and train it.
def main():
    #cuda sanity check
    print(f"CUDA available:{torch.cuda.is_available()}")
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Number of CUDA devices:{torch.cuda.device_count()}")
        print(f"Current CUDA device number:{torch.cuda.current_device()}")
        print(f"Current CUDA device name:{torch.cuda.get_device_name(torch.cuda.current_device())}")

    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


    for j in range(10):
        # a simple model, note the input layer does not need to be specified. A full list of parameters, as well as their explanantions can be seen in the main file.
        model = DyPrune(number_of_layers=4, neurons=[1000, 1000, 1000, 10],
                                         optimizer=torch.optim.AdamW,
                                         # learning_rate=0.03,
                                         pruning=False, pruning_rate="dynamic", pruning_iter=2, pruning_type="l1",
                                         pruning_min=0, pruning_max=0.05,
                                         regularization=False)
        model.to(device)
        model.forward(torch.zeros((1, 28, 28)).to(device))
        summary(model, (1, 28, 28))  # ->returns string

        train_dataloader, test_dataloader, validation_dataloader = load_data_fashion_mnist()

        model.fit(25, train_dataloader, test_dataloader, validation_dataloader)
        #model.visualize(INPUT_SIZE) #can be run to see a neural network that has been pruned. 
        model.save("results/Fashion_mnist")
    print("=====================================================================================================")

 


if __name__ == "__main__":
    main()
