from tqdm import tqdm
from models.networks import define_G
from data.custom_dataset_data_loader import GDataLoader
from util.util import tensor2im
import os
import torch
from glob import glob
from os.path import join as pjoin
from argparse import ArgumentParser
import threading
from concurrent.futures import ThreadPoolExecutor
#test
import matplotlib.pyplot as plt

from tool.inception_score import get_inception_score
from tool.getMetrics_market import ssim_score, create_masked_image

@torch.no_grad()
def get_eval_data(model, dataloader, gpu_ids=[]):
    targets = []
    generated = []
    names = []
    with tqdm(enumerate(dataloader), total=len(dataloader), desc='Generating...') as test_set:
        for i, data in test_set:
            if len(gpu_ids) > 0:
                with torch.cuda.device(gpu_ids[0]):
                    data['g_data'] = [t.cuda() for t in data['g_data']]
            output = model(data['g_data'])
            output = output.unsqueeze(1)
            target = data['target'].unsqueeze(1)
            targets.append(target.cpu())
            generated.append(output.cpu())
            names += [list(v) for v in zip(data['from'], data['to'])]
    print('Gathering results...')
    targets = torch.cat(targets, 0)
    generated = torch.cat(generated, 0)
    targets = [im for im in map(lambda t: tensor2im(t), targets)]
    generated = [im for im in map(lambda t: tensor2im(t), generated)]
    return targets, generated, names


def get_metrics(targets, generated, names, annotation_path='./market_data/market-annotation-test.csv',
                nThreads=0, nPool=0):
    masked = create_masked_image(names + names, targets + generated, annotation_path, nThreads=nPool)
    masked_targets, masked_generated = masked[:len(targets)], masked[len(targets):]
    with ThreadPoolExecutor(max_workers=1) as executor:
        IS = executor.submit(get_inception_score, generated).result()
        mask_IS = executor.submit(get_inception_score, masked_generated).result()
    SSIM = ssim_score(generated, targets, nThreads=nPool)
    mask_SSIM = ssim_score(masked_generated, masked_targets, nThreads=nPool)
    return IS[0], mask_IS[0], SSIM, mask_SSIM


def get_running_metrics(dataloader,
                        annotation_path='./market_data/market-annotation-test.csv',
                        checkpoint_dir='./checkpoints/market_PATN', gpu_ids=[], nThreads=0, nPool=0):
    pattern = '%s/[0-9]*netG.pth' % checkpoint_dir
    model_paths = glob(pattern)
    model_paths = [(int(p.split('/')[-1].split('_')[0]), p) for p in model_paths]
    metrics = []
    with tqdm(model_paths) as jobs:
        for epoch, path in jobs:
            jobs.set_description('Getting metrics for epoch %d' % epoch)
            model = define_G()
            model.load_state_dict(torch.load(path))

            if len(gpu_ids) > 0:
                with torch.cuda.device(gpu_ids[0]):
                    model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
            targets, generated, names = get_eval_data(model, dataloader, gpu_ids=gpu_ids)
            del model
            IS, mask_IS, SSIM, mask_SSIM = get_metrics(targets, generated, names, annotation_path, nThreads=nThreads, nPool=nPool)
            metrics.append((epoch, (IS, mask_IS, SSIM, mask_SSIM)))
            with open(pjoin(checkpoint_dir, 'metrics.txt'), 'a') as f:
                f.write('Epoch %d: IS %.2f mask_IS %.2f SSIM %.2f mask_SSIM %.2f\n' % (epoch, IS, mask_IS, SSIM, mask_SSIM))
        return metrics

if __name__ == '__main__':
    def device_array(s):
        try:
            return [int(v) for v in s.split(',') if int(v) >= 0]
        except:
            raise Exception('Must be comma separated integers!')
    parser = ArgumentParser('Log metrics for an experiment over multiple epochs')
    parser.add_argument('--dataroot', default='./market_data', help='Path to image and annotation data')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/market_PATN', help='Path to image and annotation data')
    parser.add_argument('--dataset', default='market', help='Which dataset (fashion or market)', choices=['market', 'fashion', 'multipie'])
    parser.add_argument('--nThreads', type=int, default=0, help='Workers for dataloader')
    parser.add_argument('--nPool', type=int, default=0, help='Workers for metrics calculation')
    parser.add_argument('--gpu_ids', type=device_array, default=[], help='GPU ids for DataParallel, -1 for CPU-only')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for model generation')
    parser.add_argument('--max_dataset_size', type=float, default=float('inf'), help='Max number of minibatches to generate')
    opt = parser.parse_args()
    if opt.dataset == 'fashion':
        pairs_file = 'fasion-resize-pairs-test.csv'
        anno_file = 'fasion-resize-annotation-test.csv'
        dataset_mode = 'keypoint'
    elif opt.dataset == 'market':
        pairs_file = 'market-pairs-test.csv'
        anno_file = 'market-annotation-test.csv'
        dataset_mode = 'keypoint'
    elif opt.dataset == 'multipie':
        pairs_file = 'multipie_pairs_test.csv'
        anno_file = 'multipie_cropped_annotations.csv'
        dataset_mode = 'multipie'

    data_loader = GDataLoader()
    data_loader.initialize(opt.dataroot,
                           'test',
                           pjoin(opt.dataroot, pairs_file),
                           opt.batch_size, True, opt.nThreads, max_dataset_size = opt.max_dataset_size, dataset_mode=dataset_mode)
    with open(pjoin(opt.checkpoint_dir, 'metrics.txt'), 'w+') as f:
        f.write('-------Experiment %s--------\n' % opt.checkpoint_dir.split('/')[-1])

    res = get_running_metrics(data_loader,
                              annotation_path=pjoin(opt.dataroot, anno_file),
                              checkpoint_dir=opt.checkpoint_dir, gpu_ids=opt.gpu_ids, nThreads=opt.nThreads, nPool=opt.nPool)
    epochs, metrics = zip(*res)
    IS, mask_IS, SSIM, mask_SSIM = zip(*metrics)
    print('saving results')
    plt.plot(epochs, IS, 'b', label='IS')
    plt.plot(epochs, mask_IS, 'g', label='mask_IS')
    plt.plot(epochs, SSIM, 'r', label='SSIM')
    plt.plot(epochs, mask_SSIM, 'm', label='mask_SSIM')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.title('Metric over epochs for experiment %s' % opt.checkpoint_dir.split('/')[-1])
    plt.legend()
    plt.savefig(pjoin(opt.checkpoint_dir,'metrics.jpg'))


