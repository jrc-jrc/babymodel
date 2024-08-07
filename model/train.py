import torch
import torch.nn as nn
import torch.optim as optim
from baby_model import AccentConversionModel
from data_loader import get_dataloader

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super(FeatureMatchingLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, real_features, fake_features):
        loss = 0
        for r_feat, f_feat in zip(real_features, fake_features):
            loss += self.l1_loss(r_feat, f_feat)
        return loss

def train(model: AccentConversionModel, train_loader, num_epochs, device):
    optimizer_G = optim.AdamW(model.parameters(), lr=0.0002, betas=(0.8, 0.99))
    optimizer_D = optim.AdamW(model.accent_discriminator.parameters(), lr=0.0002, betas=(0.8, 0.99))
    optimizer_HD = optim.AdamW(model.hifigan_discriminator.parameters(), lr=0.0002, betas=(0.8, 0.99))

    mse_loss = nn.MSELoss()
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    fm_loss = FeatureMatchingLoss()

    model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            waveform = batch['waveform'].to(device)
            mfcc = batch['mfcc'].to(device)
            f0 = batch['f0'].to(device)
            accent_id = batch['accent_id'].to(device)
            is_native = batch['is_native'].to(device)
            
            # Train Accent Discriminator
            optimizer_D.zero_grad()
            output, acoustic_encoding = model(waveform, accent_id, mfcc, f0)
            accent_pred = model.accent_discriminator(acoustic_encoding.detach())
            
            d_loss_real = bce_loss(accent_pred[is_native], torch.ones(sum(is_native), 1).to(device))
            d_loss_fake = bce_loss(accent_pred[~is_native], torch.zeros(sum(~is_native), 1).to(device))
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train HiFiGAN Discriminator
            optimizer_HD.zero_grad()
            output, _ = model(waveform, accent_id, mfcc, f0)
            real_pred, real_features = model.hifigan_discriminator(waveform)
            fake_pred, fake_features = model.hifigan_discriminator(output.detach())
            
            hd_loss_real = mse_loss(real_pred, torch.ones_like(real_pred))
            hd_loss_fake = mse_loss(fake_pred, torch.zeros_like(fake_pred))
            hd_loss = hd_loss_real + hd_loss_fake
            hd_loss.backward()
            optimizer_HD.step()

            # Train Generator (Acoustic Encoder + HiFiGAN Decoder)
            optimizer_G.zero_grad()
            output, acoustic_encoding = model(waveform, accent_id, mfcc, f0)
            
            # Reconstruction loss
            reconstruction_loss = l1_loss(output, waveform)
            
            # Adversarial loss for accent
            accent_pred = model.accent_discriminator(acoustic_encoding)
            adv_loss_accent = bce_loss(accent_pred[~is_native], torch.ones(sum(~is_native), 1).to(device))
            
            # Adversarial loss for HiFiGAN
            fake_pred, fake_features = model.hifigan_discriminator(output)
            adv_loss_hifigan = mse_loss(fake_pred, torch.ones_like(fake_pred))
            
            # Feature matching loss
            _, real_features = model.hifigan_discriminator(waveform)
            feature_matching_loss = fm_loss(real_features, fake_features)
            
            # Mel-spectrogram loss
            mel_loss = mse_loss(model.mel_spectrogram(output), model.mel_spectrogram(waveform))
            
            g_loss = (
                reconstruction_loss +
                adv_loss_accent +
                adv_loss_hifigan +
                feature_matching_loss +
                mel_loss
            )
            g_loss.backward()
            optimizer_G.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}, HD Loss: {hd_loss.item():.4f}")

    torch.save(model.state_dict(), 'accent_conversion_model.pth')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccentConversionModel(num_accents=8, input_dim=93, output_dim=512).to(device)
    train_loader = get_dataloader("path_to_your_combined_csv_file.csv")
    train(model, train_loader, num_epochs=50, device=device)