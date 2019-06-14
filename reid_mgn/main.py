import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm, trange
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter

from opt import opt
from data import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking

writer = SummaryWriter('runs/%s' % (opt.name))

class Main():
    def __init__(self, model, loss, data, optimizer=None, scheduler=None):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset
        
        if len(opt.gpu_ids) > 0:
                model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

        self.model = model.to(opt.base_device)
        self.loss = loss
        self.optimizer = optimizer if optimizer else get_optimizer(model)
        self.scheduler = scheduler if scheduler else lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=opt.lr_gamma)

    def train(self):
        self.scheduler.step()

        self.model.train()
        batches = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for batch, (inputs, labels) in batches:
            inputs = inputs.to(opt.base_device)
            labels = labels.to(opt.base_device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)

            desc = "Batch %d, triplet %.2f, x-entropy %.2f, total %.2f" % (batch, self.loss.triplet, self.loss.cross_entropy, self.loss.total)
            batches.set_description(desc)

            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        qf = extract_feature(self.model, tqdm(self.query_loader, desc='Extracting query features')).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader, desc='Extracting test features')).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        tqdm.write('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        retval = {
            're-ranking': (m_ap, r[0], r[2], r[4], r[9])
        }

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        tqdm.write('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        retval['without'] = (m_ap, r[0], r[2], r[4], r[9])

        return retval

    @torch.no_grad()
    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':

    data = Data(**opt.__dict__)
    opt.num_classes = len(data.trainset.unique_ids)
    model = MGN(**opt.__dict__)
    loss = Loss(**opt.__dict__)
    main = Main(model, loss, data)

    if opt.mode == 'train':
        if opt.continue_train:
            checkpoint = torch.load(opt.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            main.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            opt.latest_epoch = checkpoint['epoch']
            print('Continuing training from %s' % opt.checkpoint_path)
            epochs = trange(opt.latest_epoch + 1, int(opt.epoch) + 1, desc='Epoch %d' % (opt.latest_epoch + 1))
        else:
            epochs = trange(1, opt.epoch + 1, desc='Epoch 1')
        for epoch in epochs:
            # print('\nepoch', epoch)
            main.train()
            desc = "Epoch %d, triplet %.2f, x-entropy %.2f, total %.2f" % (epoch, loss.triplet, loss.cross_entropy, loss.total)
            epochs.set_description(desc)
            writer.add_scalar('Triplet loss', loss.triplet, epoch)
            writer.add_scalar('Cross Entropy loss', loss.cross_entropy, epoch)
            writer.add_scalar('Total loss', loss.total, epoch)
            if epoch % opt.eval_freq == 0:
                results = main.evaluate()
                r_map, r_r1, r_r3, r_r5, r_r10 = results['re-ranking']
                w_map, w_r1, w_r3, w_r5, w_r10 = results['without']
                writer.add_scalar('Re-ranked/mAP', r_map, epoch)
                writer.add_scalar('Re-ranked/r@1', r_r1, epoch)
                writer.add_scalar('Re-ranked/r@5', r_r5, epoch)
                writer.add_scalar('Re-ranked/r@10', r_r10, epoch)
                writer.add_scalar('Not-RR/mAP', w_map, epoch)
                writer.add_scalar('Not-RR/r@1', w_r1, epoch)
                writer.add_scalar('Not-RR/r@5', w_r5, epoch)
                writer.add_scalar('Not-RR/r@10', w_r10, epoch)
            if epoch % opt.save_freq == 0:
                os.makedirs('checkpoints/%s' % opt.name, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': main.optimizer.state_dict(),
                    'options': opt.__dict__
                }, 'checkpoints/%s/checkpoint_%d.pt' %(opt.name, epoch))
                tqdm.write('Saved checkpoint of experiment %s at epoch %d' % (opt.name, epoch))
        os.makedirs('weights/%s' % opt.name, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'options': opt.__dict__
        }, ('weights/%s/model_%d.pt' % (opt.name, opt.epoch)))
        tqdm.write('Saved weights/%s/model_%d.pt' % (opt.name, opt.epoch))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.model_path)['model_state_dict'])
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.model_path)['model_state_dict'])
        main.vis()
