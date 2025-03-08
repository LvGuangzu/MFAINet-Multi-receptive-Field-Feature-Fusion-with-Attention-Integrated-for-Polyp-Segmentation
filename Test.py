# test代码的作用就是使用我们训练好的模型将图像进行分割，得到我们预测出的mask图像
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.AAAI_use import PolypPVT
from util.dataloader import test_dataset
import cv2

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
def test_MAE(model, path, dataset):
    data_path = os.path.join(path, dataset)  # /dataset/TestDataset/CVC-300...等
    image_root = '{}/images/'.format(data_path)  # 测试集中的原始图像
    gt_root = '{}/masks/'.format(data_path)  # 测试集中的mask图像
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)  # 每张照片的大小是352*352的
    MAE = 0.0
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
        MAE = MAE + mae
    MAE_num = MAE / num1
    MAE_num = '{:.3f}'.format(MAE_num)
    MAE_num = float(MAE_num)
    return MAE_num

# 得到wfm
def test_WFM(model, path, dataset):
    data_path = os.path.join(path, dataset)  # /dataset/TestDataset/CVC-300...等
    image_root = '{}/images/'.format(data_path)  # 测试集中的原始图像
    gt_root = '{}/masks/'.format(data_path)  # 测试集中的mask图像
    model.eval()
    num1 = len(os.listdir(gt_root))
    test_loader = test_dataset(image_root, gt_root, 352)  # 每张照片的大小是352*352的
    WFM = 0.0
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
        wfm = cal_wfm()
        WFM = WFM + wfm.cal(input, target)
    WFM_num = WFM / num1
    WFM_num = '{:.3f}'.format(WFM_num)
    WFM_num = float(WFM_num)
    return WFM_num

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/PolypPVT/clinic0.945_0.899.pth')
    opt = parser.parse_args()
    model = PolypPVT()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    
    datasets = ['CVC-ClinicDB']  # Add other datasets if needed
    for dataset in datasets:
        data_path = './dataset/TestDataset/{}'.format(dataset)
        
        iou_score = test_iou(model, './dataset/TestDataset', dataset)
        dice_score = test_dice(model, './dataset/TestDataset', dataset)
        mae_score = test_MAE(model, './dataset/TestDataset', dataset)
        wfm_score = test_WFM(model, './dataset/TestDataset', dataset)
        
        print(f"Dataset: {dataset}")
        print(f"IoU: {iou_score}")
        print(f"Dice: {dice_score}")
        print(f"MAE: {mae_score}")
        print(f"WFM: {wfm_score}")
