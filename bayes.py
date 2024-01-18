"""
贝叶斯求最佳参数
"""
import argparse
from distutils.util import strtobool
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from mocovit import MoCoViT
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt.space import Real



#定义SAM优化器
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm


def objective_function(lr, momentum, weight_decay):
    model = MoCoViT()
    model.to(device)
    optimizer = SAM(model.parameters(), torch.optim.SGD, lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MoCoViT model on an ImageNet-1k dataset.')
    parser.add_argument('--imagenet_path', type=str, default='./imagenet', help="Path to ImageNet-1k directory containing 'train' and 'val' folders. Default './imagenet'.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use for training. Default 0.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for which to train. Default 20.')
    parser.add_argument('--validate', choices=('True', 'False'), default='True', help='If True, run validation after each epoch. Default True.')
    parser.add_argument('--train_batch', type=int, default=128, help='Batch size to use for training. Default 128.')
    parser.add_argument('--val_batch', type=int, default=1024, help='Batch size to use for validation. Default 1024.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use while loading dataset splits. Default 4.')
    args = parser.parse_args()
    args.validate = strtobool(args.validate)

    device = torch.device('cuda:%d' % args.gpu if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=args.num_workers)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    val_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, num_workers=args.num_workers)

    initial_lr = 0.05
    initial_momentum = 0.9
    initial_weight_decay = 1e-4

    space = [
        Real(0.001, 0.1, "log-uniform", name='lr'),
        Real(0.5, 1.0, name='momentum'),
        Real(1e-5, 1e-3, "log-uniform", name='weight_decay'),
    ]

    @use_named_args(space)
    def objective(**params):
        return objective_function(**params)

    def on_step(optim_result):
        print("Best parameters so far: {}".format(optim_result.x))

    res_gp = gp_minimize(objective, space, n_calls=50, random_state=0, callback=[on_step])
    print("Best parameters: {}".format(res_gp.x))

