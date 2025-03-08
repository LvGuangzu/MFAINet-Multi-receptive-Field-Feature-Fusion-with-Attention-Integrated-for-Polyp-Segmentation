import torch
from torch.autograd import Variable
import argparse
from datetime import datetime
from lib.MFAINetwork import MFAINet
from util.dataloader import get_loader, test_dataset
from util.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import numpy as np
import logging
import os
import matplotlib.pyplot as plt


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


# 得到IoU数据
def test_iou(model, path, dataset):
    data_path = os.path.join(path, dataset)  # /dataset/TestDataset/CVC-300...等
    image_root = '{}/images/'.format(data_path)  # 测试集中的原始图像
    gt_root = '{}/masks/'.format(data_path)  # 测试集中的mask图像
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)  # 每张照片的大小是352*352的
    
    # IOU是记录总的IoU的，每一张照片之和
    IOU = 0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        output = res
        
        target = np.array(gt)
        smooth = 1
        input_flat = np.reshape(output, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)  # 交集
        
        union = ((input_flat + target_flat) - intersection)  # 并集
        iou = (intersection.sum() + smooth) / (union.sum() + smooth)
        IOU = IOU + iou
    IOU_num = IOU / num1
    IOU_num = '{:.3f}'.format(IOU_num)
    IOU_num = float(IOU_num)
    return IOU_num


# 得到DIC这个数据
def test_dice(model, path, dataset):
    data_path = os.path.join(path, dataset)  # /dataset/TestDataset/CVC-300...等
    image_root = '{}/images/'.format(data_path)  # 测试集中的原始图像
    gt_root = '{}/masks/'.format(data_path)  # 测试集中的mask图像
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)  # 每张照片的大小是352*352的
    # DSC是记录总的dice的，每一张照片之和
    DSC = 0.0
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1 = model(image)
        # eval Dice
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        # dice = '{:.4f}'.format(dice)
        # dice = float(dice)
        DSC = DSC + dice
    DSC_num = DSC / num1
    DSC_num = '{:.3f}'.format(DSC_num)
    DSC_num = float(DSC_num)
    # 这里的DSC/num1就是mDic
    return DSC_num

# 得到MAE数据
def test_mae(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    total_mae = 0.0

    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res

        target = np.array(gt)
        mae = np.mean(np.abs(input - target))
        total_mae += mae

    mean_mae = total_mae / num1
    mean_mae = '{:.3f}'.format(mean_mae)
    mean_mae = float(mean_mae)

    return mean_mae

# 得到wfm
def test_wfm(model, path, dataset):
    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)
    total_wf = 0.0

    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1 = model(image)
        res = F.upsample(res + res1, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res

        target = np.array(gt)
        true_pos = np.sum(input * target)
        false_pos = np.sum(input * (1 - target))
        false_neg = np.sum((1 - input) * target)

        precision = true_pos / (true_pos + false_pos + 1e-8)
        recall = true_pos / (true_pos + false_neg + 1e-8)
        beta = 0.3  # 设置beta为0.3,以计算weighted F-measure
        wf = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-8)
        total_wf += wf

    mean_wf = total_wf / num1
    mean_wf = '{:.3f}'.format(mean_wf)
    mean_wf = float(mean_wf)

    return mean_wf

def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_P2_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2 = model(images)
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss = loss_P1 + loss_P2
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_P2_record.update(loss_P2.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P2_record.show()))
    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path + str(epoch) + 'MFAINet.pth')
    # choose the best model


    # 调用test_dice函数计算dice值
    test1path = './dataset/TestDataset/'
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-ClinicDB', 'Kvasir', 'ColonDB', 'ETIS-Larib', 'CVC-300']:
            dataset_dice = test_dice(model, test1path, dataset)  # 计算dice，在日志进行显示
            dataset_iou = test_iou(model, test1path, dataset)  # 计算iou，在日志进行显示
            dataset_mae = test_mae(model, test1path, dataset)  # 计算mae，在日志进行显示
            dataset_wfm = test_wfm(model, test1path, dataset)  # 计算wfm，在日志进行显示
            logging.info('epoch: {}, dataset: {}, dice: {}'.format(epoch, dataset, dataset_dice))
            logging.info('epoch: {}, dataset: {}, iou: {}'.format(epoch, dataset, dataset_iou))
            logging.info('epoch: {}, dataset: {}, mae: {}'.format(epoch, dataset, dataset_mae))
            logging.info('epoch: {}, dataset: {}, wfm: {}'.format(epoch, dataset, dataset_wfm))
            print(dataset, ' dice: ', dataset_dice)  # 在终端显示当前epoch每个数据集的dice值
            print(dataset, ' iou:', dataset_iou)  # 在终端显示当前epoch每个数据集的iou值
            print(dataset, ' mae:', dataset_mae)  # 在终端显示当前epoch每个数据集的mae值
            print(dataset, ' wfm:', dataset_wfm)  # 在终端显示当前epoch每个数据集的wfm值

if __name__ == '__main__':
    dict_plot = {'CVC-ClinicDB': [], 'Kvasir': [], 'ColonDB': [], 'ETIS-Larib': [], 'CVC-300': [], 'test': []}
    name = ['CVC-ClinicDB', 'Kvasir', 'ColonDB', 'ETIS-Larib', 'CVC-300', 'test'] 
    
    # dict_plot = {'CVC-300': [], 'test': []}
    # name = ['CVC-300', 'test'] 
    ##################model_name#############################
    model_name = 'MFAINet'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=12, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=25, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/' + model_name + '/')

    opt = parser.parse_args()
    logging.basicConfig(filename='train_log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
    model = MFAINet().cuda()

    best_dice = 0
    best_iou = 0

    params = model.parameters()

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)  # train数据中的原始图像
    gt_root = '{}/masks/'.format(opt.train_path)  # train数据中的mask图像

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
        train(train_loader, model, optimizer, epoch, opt.test_path)
    # dict_plot_dice已经在train这个函数中进行了更新
    plot_train_dice(dict_plot_dice, name)
    plot_train_iou(dict_plot_iou, name)
