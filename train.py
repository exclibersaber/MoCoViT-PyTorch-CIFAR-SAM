"""
Script to train MoCoViT on CIAFR100.
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

#准确率计算
def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k

    Source: https://github.com/pytorch/examples/blob/main/imagenet/main.py#L464
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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

    # # define train and val datasets
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # train_dataset = torchvision.datasets.ImageNet(args.imagenet_path, split='train', transform=train_transform)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch, shuffle=True, num_workers=0)

    # val_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize
    # ])
    # val_dataset = torchvision.datasets.ImageNet(args.imagenet_path, split='val', transform=val_transform)
    # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch, shuffle=False, num_workers=0)
    
    
    # CIFAR-100: 定义训练和验证数据集
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



    # load network onto gpu
    model = MoCoViT()
    model.train()
    model.to(device)

    #使用SAM优化器
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, lr=0.05, momentum=0.9, weight_decay=1e-4)
    #bayes优化参数
    # optimizer = SAM(model.parameters(), base_optimizer, lr=0.05, momentum=0.9221328742905088, weight_decay=0.0005198657849887135)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    criterion = nn.CrossEntropyLoss()

    print('Starting Training...')
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        #使用SAM
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # first step
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.first_step(zero_grad=True)          

            # second step
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.second_step(zero_grad=True)           

            running_loss += loss.item()
            if i % 500 == 0:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.6f}')
                running_loss = 0.0

        

        # save checkpoint
        torch.save(model.state_dict(), './checkpoints/epoch%s.pt' % (epoch))

        # validate
        if args.validate:
            model.eval()

            all_outputs = torch.empty((len(val_dataset), 1000))
            all_labels = torch.empty(len(val_dataset))

            print('\nValidating...')
            with torch.no_grad():
                for i, data in enumerate(val_loader):
                    inputs, labels = data[0].to(device), data[1].to(device)
                    num_labels = len(labels)
                    outputs = model(inputs)

                    offset = i * args.val_batch
                    all_outputs[offset:(offset)+num_labels] = outputs
                    all_labels[offset:(offset)+num_labels] = labels

            acc1, acc5 = accuracy(all_outputs, all_labels, topk=(1, 5))
            print("Overall  Acc@1: %.5f, Acc@5: %.5f\n" % (acc1, acc5))

    print('\nFinished Training.')
