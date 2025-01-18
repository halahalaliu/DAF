import configargparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import data_loader
import os
import torch
import models
import utils
from utils import str2bool
import numpy as np
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 修改num_workers,lr,transfer_loss_weight,transfer_loss,batch_size是否使用自适应权重,是否使用续训
# python MyDA/main.py --config MyDA/DAF/DAF.yaml --data_dir autodl-tmp/mydataset --src_domain S --tgt_domain L


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=3)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--src_domain', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=20, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")
    parser.add_argument("--resume_from", type=str, default='', help="the checkpoint file to resume from")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')
    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class


def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


def test(model, target_test_loader, epoch, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        with tqdm(total=len(target_test_loader), desc=f'Train: Epoch {epoch}/{args.n_epoch}',
                  mininterval=0.3) as pbar:
            for data, target in target_test_loader:
                
                data, target = data.to(args.device), target.to(args.device)
                s_output = model.predict(data)
                loss = criterion(s_output, target)
                test_loss.update(loss.item())
                pred = torch.max(s_output, 1)[1]
                correct += torch.sum(pred == target)
                pbar.update(1)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, tb_writer, args):
    starting_epoch = 1
    if args.resume_from:
        train_state = torch.load(args.resume_from)
        model.load_state_dict(train_state['model_state_dict'])
        optimizer.load_state_dict(train_state['optimizer_state_dict'])
        starting_epoch = train_state['epoch'] + 1
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch

    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0
    log = []
    for e in range(starting_epoch, args.n_epoch + 1):
        model.train()
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        criterion = torch.nn.CrossEntropyLoss()
        with tqdm(total=n_batch, desc=f'Train: Epoch {e}/{args.n_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for iteration in range(n_batch):
                iteration += 1
                data_source, label_source = next(iter_source)  # .next()
                data_target, _ = next(iter_target)  # .next()
                data_source, label_source = data_source.to(
                    args.device), label_source.to(args.device)
                data_target = data_target.to(args.device)

                clf_loss, transfer_loss = model(data_source, data_target, label_source)
                    

                ### 自适应损失权重 ###
                transfer_clf_weight = ((transfer_loss.item() + clf_loss.item()) * clf_loss.item()) / (
                        transfer_loss.item() ** 2 + clf_loss.item() ** 2)
                transfer_loss_weight = ((transfer_loss.item() + clf_loss.item()) * transfer_loss.item()) / (
                        transfer_loss.item() ** 2 + clf_loss.item() ** 2)

                if transfer_loss_weight and transfer_clf_weight == None:
                    loss = clf_loss + transfer_loss * args.transfer_loss_weight
                else:
                    loss = clf_loss * transfer_clf_weight + transfer_loss * transfer_loss_weight
                    args.transfer_loss_weight = transfer_loss_weight
                # loss = clf_loss + transfer_loss * args.transfer_loss_weight

                # loss = clf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()

                train_loss_clf.update(clf_loss.item())
                train_loss_transfer.update(transfer_loss.item())
                train_loss_total.update(loss.item())
                
                pbar.set_postfix(**{
                    'Clf_loss': train_loss_clf.avg,
                    'Transfer_loss': train_loss_transfer.avg,
                    'Loss': train_loss_total.avg,
                    'Lr': optimizer.param_groups[0]['lr'],
                    'transfer_loss_weight': args.transfer_loss_weight,
                })

                pbar.update(1)

        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])

        info = 'Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}'.format(
                        e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg)
        # Test
        stop += 1
        test_acc, test_loss = test(model, target_test_loader, e, args)
        info += ', test_loss {:4f}, test_acc: {:.4f}'.format(test_loss, test_acc)
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        # 记录loss acc lr
        tags = ['train_loss', 'accuracy', 'learning_rate']
        tb_writer.add_scalar(tags[0], train_loss_total.avg / n_batch, e - 1)
        tb_writer.add_scalar(tags[1], test_acc, e - 1)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]['lr'], e - 1)

        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
            checkpoint = {
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, 'tf-logs/Val_Epoch{:03}-Acc{:.3f}.pth'.format(e, test_acc))
        if args.early_stop > 0 and stop >= args.early_stop:
            print(info)
            break
        print(info)
    print('Transfer result: {:.4f}'.format(best_acc))


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)
    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    tb_writer = SummaryWriter(log_dir='tf-logs')
    # 实例化模型
    model = get_model(args)

    optimizer = get_optimizer(model, args)

    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None
    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, tb_writer, args)


if __name__ == "__main__":
    main()
