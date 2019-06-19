from tqdm import tqdm
from models.networks import define_G
from data.custom_dataset_data_loader import GDataLoader
from util.util import tensor2im
from tool.getMetrics_market import ssim_score, create_masked_image
from tool.inception_score import get_inception_score
import torch
from glob import glob
from os.path import join as pjoin
from argparse import ArgumentParser

data_loader = GDataLoader()
data_loader.initialize('/home/local/YITU-INC/stephen.ng/code/Pose-Transfer/market_data',
                       'train',
                       '/home/local/YITU-INC/stephen.ng/code/Pose-Transfer/market_data/market-pairs-train.csv',
                       8,True,10)
model = define_G()
model = model.eval()


@torch.no_grad()
def get_eval_data(model, dataloader):
    targets = None
    generated = None
    names = []
    with tqdm(enumerate(dataloader), total=len(dataloader), desc='Generating...') as test_set:
        for i, data in test_set:
            output = model(data['g_data'])
            output = output.unsqueeze(1)
            target = data['target'].unsqueeze(1)
            targets = target.cpu() if targets is None else torch.cat((targets, target.cpu()), 0)
            generated = output.cpu() if generated is None else torch.cat((generated, output.cpu()), 0)
            names += [list(v) for v in zip(data['from'], data['to'])]
    targets = [im for im in map(lambda t: tensor2im(t), targets)]
    generated = [im for im in map(lambda t: tensor2im(t), generated)]
    return targets, generated, names


def get_metrics(targets, generated, names, annotation_path='./market_data/market-annotation-test.csv'):
    masked = create_masked_image(names + names, targets + generated, annotation_path)
    masked_targets, masked_generated = masked[:len(targets)], masked[len(targets):]
    IS = get_inception_score(generated)
    mask_IS = get_inception_score(masked_generated)
    SSIM = ssim_score(generated, targets)
    mask_SSIM = ssim_score(masked_generated, masked_targets)
    return IS, mask_IS, SSIM, mask_SSIM


def get_running_metrics(dataloader,
                        annotation_path='./market_data/market-annotation-test.csv',
                        checkpoint_dir='./checkpoints/market_PATN', gpu_ids=[]):
    pattern = '%s/[0-9]*netG.pth' % checkpoint_dir
    model_paths = glob(pattern)
    model_paths = [(int(p.split('/')[-1].split('_')[0]), p) for p in model_paths]
    model = define_G()
    metrics = []
    for epoch, path in model_paths:
        try:
            model.load_state_dict(torch.load(path))
        except:
            pass

        if len(gpu_ids) > 0:
            model = torch.nn.DataParallel(model, device_id=gpu_ids)
        targets, generated, names = get_eval_data(model, dataloader)
        metrics.append((epoch, get_metrics(targets, generated, names, annotation_path)))
    return metrics

if __name__ == '__main__':
    def device_array(s):
        try:
            return [int(v) for v in s.split(',') if int(v) > 0]
        except:
            raise Exception('Must be comma separated integers!')
    parser = ArgumentParser('Log metrics for an experiment over multiple epochs')
    parser.add_argument('--dataroot', default='./market_data', help='Path to image and annotation data')
    parser.add_argument('--checkpoint_dir', default='./checkpoints/market_PATN', help='Path to image and annotation data')
    parser.add_argument('--nThreads', type=int, default=0, help='Workers for dataloader')
    parser.add_argument('--gpu_ids', type=device_array, default=[], help='GPU ids for DataParallel, -1 for CPU-only')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for model generation')
    opt = parser.parse_args()

    data_loader = GDataLoader()
    data_loader.initialize(opt.dataroot,
                           'test',
                           pjoin(opt.dataroot, 'market-pairs-test.csv'),
                           opt.batch_size, True, opt.nThreads)
    res = get_running_metrics(data_loader,
                              annotation_path=pjoin(opt.dataroot, 'market-annotation-test.csv'),
                              checkpoint_dir=opt.checkpoint_dir)
    epochs, metrics = zip(*res)
    IS, mask_IS, SSIM, mask_SSIM = zip(*metrics)
