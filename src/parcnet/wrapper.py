import torch
from .parcnet import HybridModel
from config import get_config

class PARCNetWrapper(HybridModel):
    def __init__(self, plotter, criterion, **kwargs):
        super().__init__(criterion=criterion, **kwargs)

        self.plotter = plotter
        self.validation_packets_predictions = []
        self.save_hyperparameters(get_config())
        
    def training_step(self, batch, batch_idx):
        lost, past, ar_data, _ = batch
        return super().training_step((lost, past, ar_data), batch_idx).float()
    
    def validation_step(self, batch, batch_idx):
        lost, past, ar_data, start_sample_idxs = batch
        loss, packet_loss, outputs =  super().validation_step((lost, past, ar_data), batch_idx)
    
        filtered_idxs = self.plotter.filter_batch_by_start_sample_idx(start_sample_idxs)
        if any(filtered_idxs):
            segment_idxs_mask = torch.nonzero(filtered_idxs)
            filtered_outputs = outputs[segment_idxs_mask].squeeze()
            filtered_idxs = filtered_idxs[filtered_idxs != 0].squeeze()

            filtered_idxs = filtered_idxs.unsqueeze(0) if filtered_idxs.dim() < 1 else filtered_idxs
            filtered_outputs = filtered_outputs.unsqueeze(0) if filtered_outputs.dim() < 2 else filtered_outputs
            self.validation_packets_predictions.extend(zip(filtered_outputs.tolist(), filtered_idxs.tolist()))

        return loss, packet_loss
    
    def test_step(self, batch, batch_idx):
        lost, past, ar_data, _ = batch
        return super().test_step((lost, past, ar_data), batch_idx)
    
    def on_validation_epoch_end(self) -> None:
        if self.validation_packets_predictions and self.current_epoch % 10 == 0:
            tb = self.logger.experiment
            inpaintings, xfades, start_idxs = [], [], []
            for idx, (pred, start_idx) in enumerate(self.validation_packets_predictions):
                inpaintings.append(pred[:512])
                xfades.append(pred[512:])
                start_idxs.append(start_idx)
            self.plotter.plot_artifacts(inpaintings, xfades, start_idxs, tb, "concealed", self.current_epoch)


            if self.current_epoch == 0:
                zeros = [[0] * 512 for _ in range(len(self.validation_packets_predictions))]
                zeros_xfade = [[0] * 88 for _ in range(len(self.validation_packets_predictions))]
                self.plotter.plot_artifacts(zeros, zeros_xfade, start_idxs, tb, "zeroed", self.current_epoch)
                self.plotter.plot_artifacts(zeros, zeros_xfade, start_idxs, tb, "original", self.current_epoch)
    