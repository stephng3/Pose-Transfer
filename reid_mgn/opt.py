import argparse
import torch
import os
from datetime import datetime

parser = argparse.ArgumentParser(description='reid')


parser.add_argument("--name",
                    default='unspecified',
                    help='Experiment name. Omit for generated name')

parser.add_argument('--data_path',
                    default="datasets/fashion_data",
                    help='path of dataset')

parser.add_argument('--dataset',
                    default="DeepFashion",
                    help='name of dataset')

parser.add_argument('--mode',
                    default='train', choices=['train', 'evaluate', 'vis'],
                    help='train or evaluate ')

parser.add_argument('--query_image',
                    default='0001_c1s1_001051_00.jpg',
                    help='path to the image you want to query')

parser.add_argument('--freeze',
                    default=False,
                    help='freeze backbone or not ')

parser.add_argument('--rand_erasing',
                    default=True,
                    help='Augment data with random erasing during training')

parser.add_argument('--epoch',
                    default=500,
                    type=int,
                    help='number of epoch to train until')

parser.add_argument('--triplet_loss',
                    default=1,
                    type=int,
                    help='use triplet loss')

parser.add_argument('--triplet_margin',
                    default=1.2,
                    type=float,
                    help='triplet margin between classes')

parser.add_argument('--l_triplet',
                    default=1,
                    type=float,
                    help='Weight of triplet loss')

parser.add_argument('--l_xentropy',
                    default=2,
                    type=float,
                    help='Weight of Cross Entropy loss')

parser.add_argument('--lr',
                    default=2e-4,
                    type=float,
                    help='initial learning_rate')

parser.add_argument('--lr_scheduler',
                    default=[320, 380],
                    help='Decay by lr_gamma at these epochs')

parser.add_argument('--lr_gamma',
                    default=0.1,
                    type=float,
                    help='Multiplier to decay at scheduled steps')

parser.add_argument('--optimizer',
                    default='SGD',
                    help='Type of optimizer to use.')

parser.add_argument("--batchid",
                    default=4,
                    type=int,
                    help='the batch for id')

parser.add_argument("--batchimage",
                    default=4,
                    type=int,
                    help='the batch of per id')

parser.add_argument("--batchtest",
                    default=8,
                    type=int,
                    help='the batch size for test')

parser.add_argument("--gpus",
                    default='-1',
                    help='GPUs available. -1 for CPU')

parser.add_argument("--num_workers",
                    default=0,
                    type=int,
                    help='multithreading for data loading')

parser.add_argument("--parts",
                    default='1,2,3',
                    help='partitions per branch of MGN')

parser.add_argument("--save_freq",
                    default=10,
                    type=int,
                    help='Save every n epochs')

parser.add_argument("--eval_freq",
                    default=50,
                    type=int,
                    help='evaluate every n epochs')

parser.add_argument("--continue_train",
                    action='store_true',
                    help='continue training from specified checkpoint_name')

parser.add_argument("--checkpoint_name",
                    default='latest',
                    help='Continue training from checkpoint. Omit to train from latest')

parser.add_argument('--model_name',
                    default='latest',
                    help='load weights from .pt file. latest to load largest epoch model')

parser.add_argument('--input_height',
                    default=384,
                    type=int,
                    help='height of input image into the model (multiple of 128')

parser.add_argument('--input_width',
                    default=128,
                    type=int,
                    help='width of input image into the model (multiple of 64')

opt = parser.parse_args()

str_ids = opt.gpus.split(',')
opt.gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        opt.gpu_ids.append(id)

# set gpu ids
if len(opt.gpu_ids) > 0:
    torch.cuda.set_device(opt.gpu_ids[0])
    opt.base_device = 'cuda:%d' % opt.gpu_ids[0]
else:
    opt.base_device = 'cpu'

# parse parts
opt.parts = [int(v) for v in opt.parts.split(',')]
assert len(opt.parts) == 3
assert (True, True, True) == tuple(v > 0 for v in opt.parts)

if opt.name == 'unspecified':
    opt.name = '%s_partition%s' % (opt.dataset, ''.join(map(lambda x: str(x), opt.parts)))

if opt.mode == 'train':
    if opt.continue_train:
        if opt.checkpoint_name is 'latest':
            checkpoints = filter(lambda x: x.endswith('.pt'), os.listdir('checkpoints/%s' % opt.name))
            opt.latest_epoch = max([int(chkpt.split('_')[-1].split('.')[0]) for chkpt in checkpoints])
            opt.checkpoint_path = 'checkpoints/%s/checkpoint_%d.pt' % (opt.name, opt.latest_epoch)
        else:
            opt.checkpoint_path = 'checkpoints/%s/%s' % (opt.name, opt.checkpoint_name)
            opt.latest_epoch = int(opt.checkpoint_name.split('_')[-1].split('.')[0])
    else:
        opt.latest_epoch = -1
else:
    opt.model_path = 'weights/%s/%s' % (opt.name, opt.model_name)
    if opt.model_name is 'latest':
        models = os.listdir('weights/%s' % opt.name)
        opt.latest_epoch = max([int(model.split('_')[-1].split('.')[0]) for model in models])
        opt.model_path = 'weights/%s/model_%d.pt' % (opt.name, opt.latest_epoch)
    else:
        opt.model_path = 'weights/%s/%s' % (opt.name, opt.model_name)


os.makedirs('checkpoints/%s/' % opt.name, exist_ok=True)
logfile_name = datetime.now().strftime("%Y-%m-%d_%H:%M_opt.txt")
logfile_path = 'checkpoints/%s/%s' % (opt.name, logfile_name)
with open(logfile_path, 'w+') as f:
    f.write('--------------%s----------------\n' % opt.name)
    for k,v in sorted(opt.__dict__.items()):
            f.write('%s: %s\n' % (k, v))
    f.write('------------------End Options------------------\n')
with open(logfile_path, 'r') as f:
    print(f.read())
