import os
from argparse import ArgumentParser
import torch

parser = ArgumentParser('save model from checkpoint')

parser.add_argument('--name',
                    default='Market1501_123',
                    help='Name of experiment')

parser.add_argument('--epoch',
                    default='latest',
                    help='Epoch to save from')

opt = parser.parse_args()

if opt.epoch is 'latest':
    checkpoints = filter(lambda x: x.endswith('.pt'), os.listdir('checkpoints/%s' % opt.name)) 
    opt.epoch = max([int(chkpt.split('_')[-1].split('.')[0]) for chkpt in checkpoints])
else:
    opt.epoch = int(opt.epoch)

checkpoint_path = 'checkpoints/%s/checkpoint_%d.pt' % (opt.name, opt.epoch)
checkpoint = torch.load(checkpoint_path)

os.makedirs('weights/%s' % opt.name, exist_ok=True)
model_path = 'weights/%s/model_%d.pt' % (opt.name, opt.epoch)
torch.save({
    'model_state_dict': checkpoint['model_state_dict'],
    'options': checkpoint['options']
}, model_path)
print('Saved model weights at %s' % model_path)


