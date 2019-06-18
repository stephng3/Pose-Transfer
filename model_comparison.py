import os
from os.path import join as pjoin
import ntpath
from functools import reduce
from operator import iconcat
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from util.util import tensor2im
from argparse import ArgumentParser
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from tqdm import tqdm, trange

parser = ArgumentParser('compare model outputs on test images')
parser.add_argument('--dataset', default='market')
parser.add_argument('--experiments', default='market_PATN')
parser.add_argument('--save_path', type=str, default='./results/market_meta', help='saves results here.')
parser.add_argument('--root_dir', type=str, default='./results', help='root directory of test results.')

font = ImageFont.truetype('./fonts/cmunci.ttf', size=8)

def get_experiments(experiment_names, root_dir):
    experiments = []
    for exp in experiment_names:
        test_name = os.listdir(pjoin(root_dir, exp))[0]
        full_path = pjoin(root_dir, exp, test_name, 'images')
        experiments.append((full_path, exp,))
    return experiments


def draw_text(draw, text, box):
    w, h = font.getsize(text)
    x = (box[0] + box[2] - w) // 2
    y = (box[1] + box[3] - h) // 2
    draw.text((x, y), text, font=font, fill='black')


def get_fake(path):
    vis = Image.open(path)
    return vis.crop((vis.width // 5 * 4, 0, vis.width, vis.height))


def gather_image(experiments, image_name):
    base = Image.open(pjoin(experiments[0][0], image_name))
    segment_w = base.width // 5
    res = Image.new('RGB', (segment_w * (len(experiments) + 2), base.height + 20), color='white')
    draw = ImageDraw.Draw(res)
    gt = base.crop((0, 0, segment_w, base.height + 20))
    target = base.crop((segment_w * 2, 0, segment_w * 3, base.height + 20))
    res.paste(gt, (0, 20))
    draw_text(draw, 'Gt', (0, 0, segment_w, 20))
    res.paste(target, (segment_w, 20))
    draw_text(draw, 'Pt', (segment_w, 0, segment_w * 2, 20))
    for i, (exp_dir, exp_name) in enumerate(experiments):
        path = pjoin(exp_dir, image_name)
        fake = get_fake(path)
        left = (i + 2) * segment_w
        res.paste(fake, (left, 20))
        draw_text(draw, exp_name, (left, 0, left + segment_w, 20))
    import pdb; pdb.set_trace()
    return res


opt = parser.parse_args()
experiments = get_experiments(opt.experiments.split(','), opt.root_dir)
images = os.listdir(experiments[0][0])
os.makedirs(opt.save_path, exist_ok=True)

for image in tqdm(images):
    res = gather_image(experiments, image)
    res.save(pjoin(opt.save_path, image), 'JPEG')
    import pdb; pdb.set_trace()
