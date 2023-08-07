import os
import sys
import time
import uuid
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.distributed as dist
import torch.utils.data
import tools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-c', '--cpus', default=1, type=int,
                        help='number of cpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--tracedir', default="/mnt/", type=str, )
    parser.add_argument('--sfdir', type=str, )

    args = parser.parse_args()

    os.sched_setaffinity(0, [i for i in range(args.cpus)])
    os.environ['GOMP_CPU_AFFINITY'] = "0-" + str(args.cpus - 1)
    args.world_size = args.gpus * args.nodes

    for _ in range(2):
        start = datetime.now()
        args.sharefile = str(uuid.uuid1())
        args.start = start
        mp.spawn(worker, nprocs=args.gpus, args=(args,), daemon=True)
        time.sleep(10)


class ConvNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def worker(gpu, args):
    os.sched_setaffinity(0, [i for i in range(args.cpus)])
    os.environ['GOMP_CPU_AFFINITY'] = "0-" + str(args.cpus - 1)
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl',
                            init_method='file://' + args.sfdir + args.sharefile, world_size=args.world_size,
                            rank=rank)
    torch.manual_seed(0)
    model = ConvNet()
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 1024 // args.gpus
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./',
                                              train=False,
                                              transform=transforms.ToTensor(),
                                              download=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank,
                                                                    drop_last=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False,
                                              num_workers=0)

    def evaluate(model, test_loader):
        with torch.no_grad():
            total_loss = 0.0
            model.eval()
            for i, (images, labels) in enumerate(test_loader):
                images = images.cuda(gpu)
                labels = labels.cuda(gpu)
                # Forward pass
                outputs = model(images)
                total_loss += criterion(outputs, labels)
        return total_loss

    logs = []
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        total_step = len(train_loader)
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(gpu)
            labels = labels.cuda(gpu)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('[RANK {}]: Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(rank, epoch + 1, args.epochs, i + 1,
                                                                                    total_step, loss.item()))

        total_loss = evaluate(model, test_loader)

        print(
            '[RANK {}]: test Loss: {:.4f}, time: {:5.2f}s'.format(
                rank, total_loss, (time.time() - epoch_start_time)))

        logs.append(((datetime.now() - args.start).total_seconds(), total_loss.item()))
        if total_loss <= 100:
            break


    if gpu == 0:
        tools.append_list_as_row(args.tracedir + "trace.csv",
                                 ["mnist", args.cpus, args.gpus, (datetime.now() - args.start).total_seconds()] + logs)





if __name__ == '__main__':
    main()
