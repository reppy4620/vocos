from pathlib import Path
from argparse import ArgumentParser

from torch.utils.data import DataLoader

from lightning import Trainer
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint

import config as cfg
from model import LitModule
from dataset import AudioDataset


def main(args):
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / 'ckpt'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.train.seed)

    train_ds = AudioDataset(
        file_path=args.train_file, 
        wav_dir=args.wav_dir,
        segment_size=cfg.sample_segment_size
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=12,
        pin_memory=True
    )
    valid_ds = AudioDataset(
        file_path=args.valid_file, 
        wav_dir=args.wav_dir,
        segment_size=cfg.sample_segment_size
    )
    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=12,
        pin_memory=True
    )

    if (ckpt_dir / 'last.ckpt').exists():
        lit_module = LitModule.load_from_checkpoint(ckpt_dir / 'last.ckpt', params=cfg)
    else:
        lit_module = LitModule(cfg)

    logger = CSVLogger(save_dir=out_dir, name='logs', version=1)
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        every_n_epochs=cfg.train.save_ckpt_interval,
        save_last=True,
    )

    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.num_epoch,
        callbacks=[ckpt_callback]
    )
    trainer.fit(model=lit_module, train_dataloaders=train_dl, val_dataloaders=valid_dl)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--valid_file', type=str, required=True)
    parser.add_argument('--wav_dir',    type=str, required=True)
    parser.add_argument('--lab_dir',    type=str, required=True)
    parser.add_argument('--out_dir',    type=str, required=True)
    args = parser.parse_args()
    main(args)
