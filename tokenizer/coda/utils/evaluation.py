import numpy as np
from tqdm import tqdm
from PIL import Image
from pytorch_fid.inception import InceptionV3

import torch

from .utils_image import (
    calculate_psnr,
    calculate_ssim,
)

class FIDCalculator:
    def __init__(self, accelerator, n_samples, test_bsz):
        # feature_extractor = FeatureExtractorInceptionV3(
        #     name='inception-v3',
        #     features_list=['2048', 'logits_unbiased'],
        #     feature_extractor_internal_dtype='float32'
        # ).to(device)
        feature_extractor = InceptionV3().to(accelerator.device)
        feature_extractor.eval()

        self.inception = feature_extractor
        self.n_samples = n_samples
        self.device = accelerator.device

        batch_size = test_bsz * accelerator.num_processes
        self.batch_size = batch_size
        self.total_iters = len(self.amortize(self.n_samples, batch_size))
        self.has_init = False

    def init(self):
        self.pred_tensor = torch.empty((self.n_samples + self.batch_size, 2048), device=self.device)
        self.gt_tensor = torch.empty((self.n_samples + self.batch_size, 2048), device=self.device)
        self.ssim_tensor = torch.empty((self.n_samples + self.batch_size), device=self.device)
        self.psnr_tensor = torch.empty((self.n_samples + self.batch_size), device=self.device)
        self.idx = 0
        self.pbar = tqdm(
            total=self.total_iters, desc='FID', leave=True,
        )
        self.has_init = True

    @staticmethod
    def convert_img(img_tensor):
        img_tensor = img_tensor * 255
        img_tensor = img_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype('uint8')
        return Image.fromarray(img_tensor)

    @staticmethod
    def amortize(n_samples, batch_size):
        k = n_samples // batch_size
        r = n_samples % batch_size
        return k * [batch_size] if r == 0 else k * [batch_size] + [r]

    @staticmethod
    def calc_fid(pred_tensor, gt_tensor):
        m1 = torch.mean(pred_tensor, dim=0)
        pred_centered = pred_tensor - pred_tensor.mean(dim=0)
        s1 = torch.mm(pred_centered.T, pred_centered) / (pred_tensor.size(0) - 1)

        m2 = torch.mean(gt_tensor, dim=0)
        gt_centered = gt_tensor - gt_tensor.mean(dim=0)
        s2 = torch.mm(gt_centered.T, gt_centered) / (gt_tensor.size(0) - 1)

        a = (m1 - m2).square().sum(dim=-1)
        b = s1.trace() + s2.trace()
        c = torch.linalg.eigvals(s1 @ s2).sqrt().real.sum(dim=-1)

        _fid = (a + b - 2 * c).item()
        return _fid

    def get_metrics(self, accelerator):
        pred_tensor = self.collate_tensor(self.pred_tensor, accelerator)
        gt_tensor = self.collate_tensor(self.gt_tensor, accelerator)
        fid = self.calc_fid(pred_tensor, gt_tensor)

        ssim_tensor = self.collate_tensor(self.ssim_tensor, accelerator)
        psnr_tensor = self.collate_tensor(self.psnr_tensor, accelerator)

        self.has_init = False
        return {
            f'fid_{self.n_samples}': fid,
            'ssim': torch.mean(ssim_tensor).item(),
            'psnr': torch.mean(psnr_tensor).item(),
        }

    def collate_tensor(self, tensor, accelerator):
        tensor = tensor[:self.idx]
        # gather
        all_tensor = accelerator.gather(tensor)
        assert self.n_samples <= all_tensor.shape[0]
        all_tensor = all_tensor[:self.n_samples]
        return all_tensor

    def add(self, accelerator, samples, gt):
        if not self.has_init:
            self.init()
        samples = samples * 0.5 + 0.5
        gt = gt * 0.5 + 0.5
        samples = samples.clamp_(0., 1.)
        gt = gt.clamp_(0., 1.)

        ssim_list, psnr_list = [], []
        for (img1, img2) in zip(samples, gt):
            pil_img1 = np.array(self.convert_img(img1.squeeze()))
            pil_img2 = np.array(self.convert_img(img2.squeeze()))

            ssim_list.append(torch.tensor(calculate_ssim(pil_img1, pil_img2)))
            psnr_list.append(torch.tensor(calculate_psnr(pil_img1, pil_img2)))
        ssim_tensor = torch.stack(ssim_list)
        psnr_tensor = torch.stack(psnr_list)

        features_2048 = self.inception(samples.float())[0]
        gt_2048 = self.inception(gt.float())[0]

        bs = features_2048.shape[0]
        features_2048 = features_2048.view(bs, -1)
        gt_2048 = gt_2048.view(bs, -1)

        self.pred_tensor[self.idx:self.idx + bs] = features_2048
        self.gt_tensor[self.idx:self.idx + bs] = gt_2048
        self.ssim_tensor[self.idx:self.idx + bs] = ssim_tensor
        self.psnr_tensor[self.idx:self.idx + bs] = psnr_tensor

        self.idx = self.idx + bs
        self.pbar.update(1)

        if self.pbar.n == self.total_iters:
            return self.get_metrics(accelerator)