import torch
import torch.nn as nn
import os
import argparse
from dataset import get_dataloader
from tqdm import tqdm
from network import SODNet
from evaluation.eval_from_imgs import eval_metric
from torchvision import transforms


def main(args):
    device = torch.device("cuda")
    model = SODNet()
    model = model.to(device)
    model_path = os.path.join(args.ckpt_root, args.ckpt_name)
    evaluate(args, model=model, model_path=model_path, device=device)


def evaluate(args, model, model_path, device):
    model_dict = torch.load(model_path, weights_only=True)
    model_name = model_path.rsplit('/', 1)[-1]
    print('Loaded', model_name)

    model.to(device)
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    for testset in args.eval_datasets:
        if args.corruption:
            test_img_path = os.path.join(os.path.join(args.data_root, testset), "Corruption")
        else:
            test_img_path = os.path.join(os.path.join(args.data_root, testset), "Image")
        test_gt_path = os.path.join(os.path.join(args.data_root, testset), "GroundTruth")
        pred_root = os.path.join(os.path.join(args.pred_root, testset), args.method_name)

        test_loader = get_dataloader(
            test_img_path, test_gt_path, args.size, 1, shuffle=False, num_workers=10, pin=True)

        for batch in tqdm(test_loader, desc='Generating images from {}'.format(testset)):
            inputs = batch[0].to(device).squeeze(0)
            gts = batch[1].to(device).squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]
            with torch.no_grad():
                scaled_preds = model(inputs)[-1].sigmoid()
            os.makedirs(os.path.join(pred_root, subpaths[0][0].split('/')[0]), exist_ok=True)
            num = gts.shape[0]
            for inum in range(num):
                subpath = subpaths[inum][0]
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                res = nn.functional.interpolate(scaled_preds[inum].unsqueeze(0), size=ori_size, mode='bilinear',
                                                align_corners=True)
                save_tensor_img(res, os.path.join(pred_root, subpath))
    torch.cuda.empty_cache()
    eval_metric(args)


def save_tensor_img(tenor_im, path):
    im = tenor_im.cpu().clone()
    im = im.squeeze(0)
    tensor2pil = transforms.ToPILImage()
    im = tensor2pil(im)
    im.save(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SCNet')
    parser.add_argument('--size',
                        default=256,
                        type=int,
                        help='input size')
    parser.add_argument('--method_name', default='Ours_D+S', type=str, help='model folder')
    parser.add_argument('--ckpt_name', default='Best_DS.pth', type=str, help='pth name, None')
    parser.add_argument('--corruption', default=False, type=str, help='corruption test')
    parser.add_argument('--ckpt_root', default='./', type=str, help='model folder')
    parser.add_argument('--pred_root', default='./pred_dir', type=str, help='Output folder')
    parser.add_argument('--data_root', type=str, default=r'./test_data/test')
    parser.add_argument('--eval_datasets', nargs='+',
                        default=['CoCA', 'CoSal2015', 'CoSOD3k'])
    args = parser.parse_args()
    main(args)
