import torch
import torch.optim as optim
import torch.nn.functional as F

from lightning import LightningModule

from audio import SpectrogramTransform
from .modules.vocos import Vocos
from .modules.discriminator import Discriminator
from .loss import (
    discriminator_loss, 
    generator_loss, 
    feature_loss, 
)


class LitModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loss_coef = params.loss_coef
        self.segment_size = params.segment_size
        self.sample_segment_size = params.sample_segment_size
        self.hop_length = params.hop_length

        self.automatic_optimization = False

        self.net_g = Vocos(**params.vocos)
        self.net_d = Discriminator()

        self.spec_tfm = SpectrogramTransform(**params.audio)
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.net_g(mel).squeeze(1).cpu()
    
    def infer_gt(self, wav: torch.Tensor):
        spec = self.spec_tfm.to_spec(wav)
        o = self.net_g.infer_gt(spec)
        return o.cpu().squeeze(1)
    
    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()
        
        _, y = batch
        y_mel = self.spec_tfm.to_mel(y.squeeze(1))
        y_hat = self.net_g(y_mel)
        y_hat_mel = self.spec_tfm.to_mel(y_hat.squeeze(1))
        
        d_real, d_fake, _, _ = self.net_d(y, y_hat.detach())
        loss_d = discriminator_loss(d_real, d_fake)
        if train:
            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()

        _, d_fake, fmap_real, fmap_fake = self.net_d(y, y_hat)
        loss_gen = generator_loss(d_fake)
        loss_mel = self.loss_coef.mel * F.l1_loss(y_hat_mel, y_mel)
        loss_fm  = self.loss_coef.fm * feature_loss(fmap_real, fmap_fake)
        loss_g = (
            loss_gen +
            loss_mel +
            loss_fm
        )
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = {
            'disc': loss_d,
            'gen': loss_gen,
            'mel': loss_mel,
            'fm': loss_fm
        }
        
        self.log_dict(loss_dict, prog_bar=True)
    
    def training_step(self, batch):
        self._handle_batch(batch, train=True)

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch, batch_idx):
        self._handle_batch(batch, train=False)

    def configure_optimizers(self):
        optimizer_g = optim.AdamW(self.net_g.parameters(), **self.params.optimizer_g)
        optimizer_d = optim.AdamW(self.net_d.parameters(), **self.params.optimizer_d)
        scheduler_g = optim.lr_scheduler.ExponentialLR(optimizer_g, **self.params.scheduler_g)
        scheduler_d = optim.lr_scheduler.ExponentialLR(optimizer_d, **self.params.scheduler_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
