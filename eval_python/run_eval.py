import os
import cv2
from tqdm import tqdm
import metrics as M
import matplotlib.pyplot as plt


def cal_metrics(ds_name, gt_dir, pred_dir):

    FM = M.Fmeasure()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    gt_name_list = sorted(os.listdir(pred_dir))

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):

        gt_path = os.path.join(gt_dir, gt_name, 'GT')
        pred_path = os.path.join(pred_dir, gt_name, 'Pred')

        gt_img_list = sorted(os.listdir(pred_path))

        for i in gt_img_list:

            gt_fp = os.path.join(gt_path, i)
            pred_fp = os.path.join(pred_path, i)

            gt = cv2.imread(gt_fp, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_fp, cv2.IMREAD_GRAYSCALE)
            FM.step(pred=pred, gt=gt)
            WFM.step(pred=pred, gt=gt)
            SM.step(pred=pred, gt=gt)
            EM.step(pred=pred, gt=gt)
            MAE.step(pred=pred, gt=gt)

    fm = FM.get_results().get('fm')
    wfm = WFM.get_results().get('wfm')
    sm = SM.get_results().get('sm')
    em = EM.get_results().get('em')
    mae = MAE.get_results().get('mae')

    metrics = {
        'Smeasure': sm.round(4),
        'wFmeasure': wfm.round(4),
        'MAE': mae.round(4),
        'adpEm': em['adp'].round(4),
        'meanEm': '-' if em['curve'] is None else em['curve'].mean().round(4),
        'maxEm': '-' if em['curve'] is None else em['curve'].max().round(4),
        'adpFm': fm['adp'].round(4),
        'meanFm': fm['curve'].mean().round(4),
        'maxFm': fm['curve'].max().round(4),
    }

    metrics = {f'{k}': v for k, v in metrics.items()}

    return metrics


if __name__ == '__main__':

    metrics = cal_metrics(
        'MoCA',
        '/home/fandengping01/shuowang_project/SLT-Net/TestDataset_per_sq',
        '/home/fandengping01/shuowang_project/sltnet_om_res_ms',
    )

    print(metrics)
