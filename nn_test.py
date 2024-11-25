'''
how to we backpropagate the gradient?
can we learn a specific activation for specific task and model (layer)?
'''

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import numpy as np
import random
import src.registry as registry
import os
import time
from tqdm import tqdm
import argparse


def get_activation_fn(activation: str, *args, **kwargs):
    try:
        return registry.objective_function.build_func(activation, *args, **kwargs)
    except ValueError:
        try:
            return registry.interpolator.from_pretrained(activation, *args, **kwargs)
        except ValueError:
            raise NotImplementedError("Activation function {} not implemented".format(activation))

    
class VGG11(torch.nn.Module):
    vgg11_act_dict = {
        "features": [1,4,7,9,12,14,17,19],
        "classifier": [1,4],
    }
    out_features = 1000
    # warning: two methods both may cause small output and hard to step
    def __init__(self, activation_fn_config, num_classes=10) -> None:
        super().__init__()
        self.model = torchvision.models.vgg11(weights=None)
        # self.trailing = torch.nn.Sequential(
        #     get_activation_fn(**activation_fn_config),
        #     torch.nn.Linear(self.out_features, num_classes),
        # )
        for layer, idx in self.vgg11_act_dict.items():
            for i in idx:
                # will have different instance each call
                self.model.__getattr__(layer)[i] = get_activation_fn(**activation_fn_config)
        in_features = 4096
        self.model.__getattr__("classifier")[-1] = torch.nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        # x = self.trailing(x)
        return x

# use this one for now
class ResNet18(torch.nn.Module):
    out_features = 1000
    def __init__(self, num_classes=10, *args, **kwargs) -> None:
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        self.trailing = torch.nn.Linear(self.out_features, num_classes)
        self.apply(subsitute_module(torch.nn.ReLU, get_activation_fn, *args, **kwargs))
    
    def forward(self, x):
        x = self.model(x)
        x = self.trailing(x)
        return x

def subsitute_module(old_module_class, new_module_class, *args, **kwargs):
    def wrapper(module):
        for n, m in module.named_children():
            if isinstance(m, old_module_class):
                setattr(module, n, new_module_class(*args, **kwargs))
    return wrapper

# register loss and check acc every epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_acc = 0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss_acc += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss_acc

def test_epoch(model, dataloader, criterion, device):
    model.eval()
    loss_acc = 0
    acc_acc = 0
    for x, y in tqdm(dataloader):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        loss_acc += loss.item()
        acc_acc += (pred.argmax(dim=1) == y).sum().item()
    return loss_acc, acc_acc

def cifar10_test(model, logging_dir, dataset_dir, optimizer_class, criterion_class, lr=0.001, epoch=100, batch_size=128, device="cpu"):
    logging_dir = os.path.join(logging_dir, "cifar10")
    os.makedirs(logging_dir, exist_ok=True)
    dataset_dir = os.path.join(dataset_dir, "cifar10")
    os.makedirs(dataset_dir, exist_ok=True)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    dataset_train = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform)
    dataset_test = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    model.to(device)
    optimizer = optimizer_class(model.parameters(), lr=lr)
    criterion = criterion_class()

    file = open(os.path.join(logging_dir, "metric.txt"), "w")
    writer = SummaryWriter(os.path.join(logging_dir, "tensorboard"))

    start_time = time.time()
    best_epoch = 0
    best_loss = 1e10
    params = model.state_dict()
    for i in range(epoch):
        print("epoch {}".format(i))
        train_loss = train_epoch(model, dataloader_train, criterion, optimizer, device)
        test_loss, test_acc = test_epoch(model, dataloader_test, criterion, device)
        test_acc = test_acc / len(dataset_test)
        writer.add_scalar("train_loss", train_loss, i)
        writer.add_scalar("test_loss", test_loss, i)
        writer.add_scalar("test_acc", test_acc, i)
        print("test loss: {}, test acc: {}".format(test_loss, test_acc))
        if test_loss < best_loss:
            params = model.state_dict()
            best_loss = test_loss
            best_epoch = i
    torch.save(params, os.path.join(logging_dir, "model_{}.pth".format(best_epoch)))

    time_used = time.time() - start_time
    file.write("time used: {}\n".format(time_used))
    file.write("average time per epoch: {}".format(time_used / epoch))

    file.close()
    writer.close()

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.manual_seed(seed + i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    # model
    parser.add_argument("--activation", choices=["relu", "sigmoid", "tanh", 
                                                 "equidistant", "equidistant_tuned_values", "adaptive", "adaptive_tuned_values"], default="sigmoid")
    parser.add_argument("--path", type=str, default="./logs/exp/model.pth")
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--bl", type=float, default=-10)
    parser.add_argument("--br", type=float, default=10)
    parser.add_argument("--sl", type=float, default=0)
    parser.add_argument("--sr", type=float, default=0)
    parser.add_argument("--min_val", type=float, default=None)
    parser.add_argument("--max_val", type=float, default=None)
    # TODO: currently training is not supported
    parser.add_argument("--freeze", action="store_true", default=False)
    # image-test
    parser.add_argument("--image-test", action="store_true", default=False)
    parser.add_argument("--dataset", choices=["cifar10"], default="cifar10") # TODO: add more dataset
    parser.add_argument("--model", choices=["vgg11", "resnet18"], default="resnet18") # TODO: add more model
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--optim", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--loss_fn", choices=["cross_entropy"], default="cross_entropy")
    parser.add_argument("--device", type=str, default="cpu")
    # logging
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--logging_root", type=str, default="./test")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = arg_parse()
    seed_everything(args.seed)
    logging_dir = os.path.join(args.logging_root, args.exp_name)
    dataset_dir = os.path.join(args.logging_root, "dataset")
    activation_fn_config = {
        "activation": args.activation,
        "path": args.path,
        "N": args.N,
        "degree": args.degree,
        "bl": args.bl,
        "br": args.br,
        "sl": args.sl,
        "sr": args.sr,
        "min_val": args.min_val,
        "max_val": args.max_val,
        "freeze": args.freeze,
        "device": args.device,
    }

    optimizer_class = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    optimizer_class = optimizer_class[args.optim]
    criterion_class = {
        "cross_entropy": torch.nn.CrossEntropyLoss,
    }
    criterion_class = criterion_class[args.loss_fn]
    model_class = {
        "vgg11": VGG11,
        "resnet18": ResNet18,
    }
    model = model_class[args.model](**activation_fn_config)
    model.to(args.device)
    cifar10_test(model, logging_dir=logging_dir, dataset_dir=dataset_dir, 
                 optimizer_class=optimizer_class, criterion_class=criterion_class, 
                 lr=args.lr, epoch=args.epoch, batch_size=args.batch_size, device=args.device)
