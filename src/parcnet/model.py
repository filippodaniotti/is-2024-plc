"""
https://github.com/polimi-ispl/PARCnet/blob/main/parcnet/model.py
"""
import numpy as np
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F

def nmse(y_pred, y_true) -> np.ndarray:
    assert type(y_pred) == type(
        y_true), f'y_pred and y_true must be of the same type. Found {type(y_pred)} (y_pred) and {type(y_true)} (y_true).'

    if isinstance(y_pred, torch.Tensor):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

    nmse_db = 20 * np.log10(np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true))

    return nmse_db

def mean_absolute_error(preds, targets):
    return (preds - targets).abs().sum(axis=1).mean()

def mean_squared_error(preds, targets):
    return (preds - targets).pow(2).sum(axis=1).mean()


class DilatedResBlock(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int, alpha: float = 0.2):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(input_channel, output_channel, kernel_size, padding=2 * (kernel_size - 1) // 2, dilation=2),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(alpha, inplace=True)
        )

        self.conv_2 = nn.Sequential(
            nn.Conv1d(output_channel, output_channel, kernel_size, padding=4 * (kernel_size - 1) // 2, dilation=4),
            nn.BatchNorm1d(output_channel),
            nn.LeakyReLU(alpha, inplace=True)
        )

        self.in_conv = nn.Conv1d(input_channel, output_channel, kernel_size, padding=kernel_size // 2)

    def forward(self, inputs):
        skip = self.in_conv(inputs)
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = x + skip
        return x


class DownSample(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.downsample = nn.MaxPool1d(factor, factor)

    def forward(self, inputs):
        return self.downsample(inputs)


class UpSample(nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=factor, mode='linear', align_corners=True)

    def forward(self, inputs):
        return self.upsample(inputs)


class GLUBlock(nn.Module):
    def __init__(self, n_channels: int, dilation_rate: int):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv1d(n_channels, n_channels // 2, kernel_size=1, dilation=1),
            nn.BatchNorm1d(n_channels // 2)
        )

        self.padding = nn.ConstantPad1d((int(dilation_rate * 10), 0), value=0.)

        self.conv_left = nn.Sequential(
            nn.PReLU(),
            self.padding,
            nn.Conv1d(n_channels // 2, n_channels // 2, kernel_size=11, dilation=dilation_rate),
            nn.BatchNorm1d(n_channels // 2)
        )

        self.conv_right = nn.Sequential(
            nn.PReLU(),
            self.padding,
            nn.Conv1d(n_channels // 2, n_channels // 2, kernel_size=11, dilation=dilation_rate),
            nn.BatchNorm1d(n_channels // 2),

        )

        self.out_conv = nn.Sequential(
            nn.Conv1d(n_channels // 2, n_channels, kernel_size=1, dilation=1),
            nn.BatchNorm1d(n_channels)
        )

        self.out_prelu = nn.PReLU()

    def forward(self, inputs):
        x = self.in_conv(inputs)
        xl = self.conv_left(x)
        xr = self.conv_right(x)
        x = xl * torch.sigmoid(xr)
        x = self.out_conv(x)
        x = self.out_prelu(x + inputs)
        return x


class Generator(nn.Module):
    def __init__(self, channels: int = 1, lite: bool = True):
        super().__init__()
        self.channels = channels
        self.init_dim = 8 if lite else 16

        self.body = nn.Sequential(
            DilatedResBlock(self.channels, self.init_dim, 11),
            DownSample(),
            DilatedResBlock(self.init_dim, self.init_dim * 2, 11),
            DownSample(),
            DilatedResBlock(self.init_dim * 2, self.init_dim * 4, 11),
            DownSample(),
            DilatedResBlock(self.init_dim * 4, self.init_dim * 8, 11),
            DownSample(),
            GLUBlock(dilation_rate=1, n_channels=self.init_dim * 8),
            GLUBlock(dilation_rate=2, n_channels=self.init_dim * 8),
            GLUBlock(dilation_rate=4, n_channels=self.init_dim * 8),
            GLUBlock(dilation_rate=8, n_channels=self.init_dim * 8),
            GLUBlock(dilation_rate=16, n_channels=self.init_dim * 8),
            GLUBlock(dilation_rate=32, n_channels=self.init_dim * 8),
            UpSample(),
            DilatedResBlock(self.init_dim * 8, self.init_dim * 8, 7),
            UpSample(),
            DilatedResBlock(self.init_dim * 8, self.init_dim * 4, 7),
            UpSample(),
            DilatedResBlock(self.init_dim * 4, self.init_dim * 2, 7),
            UpSample(),
            DilatedResBlock(self.init_dim * 2, self.init_dim, 7)
        )

        self.last_conv = nn.Sequential(
            nn.ConvTranspose1d(self.init_dim, 1, 1),
            nn.Tanh(),
        )

    def forward(self, inputs):
        x = self.body(inputs)
        x = self.last_conv(x)
        return x



class HybridModel(pl.LightningModule):
    def __init__(
        self, 
        channels: int, 
        lite: bool, 
        criterion: nn.Module,
        packet_dim: int = 320, 
        extra_pred_dim: int = 80, 
        lmbda: float = 100., 
        **kwargs
    ):
        super().__init__()
        self.kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}
        self.generator = Generator(channels=channels, lite=lite)
        self.lmbda = 100.0
        self.mse_loss = F.mse_loss
        self.criterion = criterion
        self.packet_dim = packet_dim
        self.pred_dim = packet_dim + extra_pred_dim

    def configure_optimizers(self):
        optimizer_g = torch.optim.RAdam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g)
        return {"optimizer": optimizer_g, "lr_scheduler": scheduler, "monitor": "packet_val_loss"}

    def forward(self, x):
        return self.generator(x).squeeze()[..., -600:]

    def training_step(self, batch, batch_idx):
        wav, past, ar_data = batch
        pred = self.forward(past) + ar_data
        
        tot_loss = self.criterion(
            torch.cat((past.squeeze()[..., :-600], pred), dim=-1), 
            torch.cat((past.squeeze()[..., :-600], wav), dim=-1))

        self.log('tot_loss', tot_loss, prog_bar=True)

        return tot_loss

    def validation_step(self, batch, batch_idx):
        wav, past, ar_data = batch
        pred = self.forward(past) + ar_data
        # pred = ar_data

        val_loss = nmse(y_pred=pred, y_true=wav)
        packet_val_loss = nmse(y_pred=pred[..., -self.pred_dim:], y_true=wav[..., -self.pred_dim:])

        self.log('val_loss', val_loss)
        self.log('packet_val_loss', packet_val_loss)

        self.log('val_mae', mean_absolute_error(pred[..., -self.packet_dim:], wav[..., -self.packet_dim:]))

        return val_loss, packet_val_loss, pred

    def test_step(self, batch, batch_idx):
        wav, past, ar_data = batch

        pred = self.forward(past) + ar_data

        loss = nmse(y_pred=pred[..., -self.packet_dim:], y_true=wav[..., -self.packet_dim:], )

        self.log('test_loss', loss)
        self.log('test_mae', mean_absolute_error(pred[..., -self.packet_dim:], wav[..., -self.packet_dim:]))

        return loss