class DotDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


train = DotDict(
    seed=42,
    num_epoch=1000,
    batch_size=16,
    save_ckpt_interval=500,
)

loss_coef = DotDict(
    mel=45,
    fm=2
)

mel_dim = 80
sample_rate = 24000
n_fft = 1024
hop_length = 256
segment_size = 64
sample_segment_size = segment_size * hop_length

audio = DotDict(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=n_fft,
    hop_length=hop_length,
    power=1,
    f_min=0.0,
    f_max=sample_rate // 2,
    n_mels=mel_dim,
    mel_scale='slaney',
    norm='slaney',
    center=False
)

vocos = DotDict(
    in_channel=mel_dim, 
    channel=512,
    h_channel=1536,
    out_channel=n_fft + 2,
    num_layers=8,
    istft_config=DotDict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        center=True,
    )
)

optimizer_g = DotDict(
    lr=2e-4,
    betas=(0.8, 0.99),
)
optimizer_d = DotDict(
    lr=2e-4,
    betas=(0.8, 0.99),
)

scheduler_g = DotDict(
    gamma=0.999
)
scheduler_d = DotDict(
    gamma=0.999
)
