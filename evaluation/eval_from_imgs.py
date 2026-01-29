import os
import cv2
import evaluation.metric as M
from datetime import datetime
from tqdm import tqdm


def get_metric_function():
    return {
        'FM': M.Fmeasure_and_FNR(),
        'WFM': M.WeightedFmeasure(),
        'SM': M.Smeasure(),
        'EM': M.Emeasure(),
        'MAE': M.MAE()
    }


def eval_metric(args):
    model_name = args.method_name
    dataset_list = args.eval_datasets
    data_root = args.data_root
    pred_root = args.pred_root

    for i in range(len(dataset_list)):
        dataset = dataset_list[i]
        pred_data_dir = os.path.join(pred_root, dataset)
        label_data_dir = os.path.join(data_root, dataset, 'GroundTruth')
        mertic_fun = get_metric_function()

        classes = os.listdir(label_data_dir)
        for k in tqdm(range(len(classes)), desc='Model {} Evaluating on {} dataset'.format(model_name, dataset)):
            class_name = classes[k]
            img_list = os.listdir(os.path.join(label_data_dir, class_name))
            for l in range(len(img_list)):
                img_name = img_list[l]
                pred = cv2.imread(os.path.join(pred_data_dir, model_name, class_name, img_name), 0)
                gt = cv2.imread(os.path.join(label_data_dir, class_name, img_name[:-4] + '.png'), 0)
                for _, fun in mertic_fun.items():
                    fun.step(pred=pred / 255, gt=gt / 255)

        fm = mertic_fun['FM'].get_results()[0]['fm']
        wfm = mertic_fun['WFM'].get_results()['wfm']
        sm = mertic_fun['SM'].get_results()['sm']
        em = mertic_fun['EM'].get_results()['em']
        mae = mertic_fun['MAE'].get_results()['mae']
        fnr = mertic_fun['FM'].get_results()[1]
        time = datetime.now().strftime("%Y.%m.%d %H:%M")
        eval_res = '{} Method {}: \nSmeasure:{:.4f} || meanEm:{:.4f} || adpEm:{:.4f} || maxEm:{:.4f} || wFmeasure:{:.4f} || ' \
                   'adpFm:{:.4f} || meanFm:{:.4f} || maxFm:{:.4f} ||  MAE:{:.4f} || fnr:{:.4f}'.format(
            time, model_name, sm, em['curve'].mean(), em['adp'], em['curve'].max(), wfm, fm['adp'],
            fm['curve'].mean(), fm['curve'].max(), mae, fnr)
        print(eval_res)
