import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model
from .pronunciation_encoder import PronunciationEncoder
from .acoustic_encoder import AcousticEncoder
from .hifigan_decoder import HiFiGANDecoder
from .accent_discriminator import AccentDiscriminator

class AccentConversionModel(nn.Module):
    def __init__(self, num_accents, input_dim, output_dim):
        super(AccentConversionModel, self).__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")
        self.pronunciation_encoder = PronunciationEncoder(num_accents)
        self.acoustic_encoder = AcousticEncoder(input_dim)
        self.hifigan_decoder = HiFiGANDecoder(output_dim, [8,8,2,2], [16,16,4,4])
        self.accent_discriminator = AccentDiscriminator(256)

        # Mel-spectrogram converter
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=256, n_mels=80
        )

    def forward(self, waveform, accent_id, mfcc, f0):
        # Extract wav2vec features (text predictions)
        wav2vec_output = self.wav2vec(waveform).last_hidden_state
        # Pass through pronunciation encoder
        pronunciation_encoding = self.pronunciation_encoder(wav2vec_output, accent_id)
        # Pass through acoustic encoder
        acoustic_encoding = self.acoustic_encoder(torch.cat([mfcc, f0], dim=-1))
        # Combine features
        combined_features = torch.cat([
            pronunciation_encoding, 
            acoustic_encoding.unsqueeze(1).expand(-1, pronunciation_encoding.size(1), -1),
            f0.unsqueeze(-1).expand(-1, pronunciation_encoding.size(1), -1)
        ], dim=-1)
        # Generate output through HiFiGAN decoder
        output = self.hifigan_decoder(combined_features.transpose(1, 2))
        
        return output, acoustic_encoding

    def discriminate_accent(self, acoustic_encoding):
        return self.accent_discriminator(acoustic_encoding)

    def compute_mel_spectrogram(self, audio):
        return self.mel_spectrogram(audio)