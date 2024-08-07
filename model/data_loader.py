import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor
import numpy as np
import librosa

class AccentDataset(Dataset):
    def __init__(self, csv_file, max_audio_length=160000):
        self.data = pd.read_csv(csv_file)
        self.max_audio_length = max_audio_length
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.accent_to_id_dict = {accent: idx for idx, accent in enumerate(self.data['accent'].unique())}
        self.native_accents = ['AM']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and preprocess audio
        waveform, sample_rate = torchaudio.load(row['file_path'])
        waveform = waveform.squeeze(0)  # Remove channel dim
        
        # Pad or trim the audio to max_audio_length
        if waveform.size(0) < self.max_audio_length:
            waveform = torch.nn.functional.pad(waveform, (0, self.max_audio_length - waveform.size(0)))
        else:
            waveform = waveform[:self.max_audio_length]

        # Extract MFCC features
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)(waveform)

        # Extract F0 (fundamental frequency) using YAAPT algorithm
        # Note: You'll need to implement or use a library for YAAPT
        f0 = self.extract_f0(waveform.numpy(), sample_rate)

        # Get wav2vec features
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        
        # Get accent ID
        accent_id = self.accent_to_id_dict[row['accent']]
        is_native = row['accent'] in self.native_accents

        return {
            'waveform': waveform,
            'mfcc': mfcc,
            'f0': torch.tensor(f0).float(),
            'wav2vec_input': inputs.input_values,
            'accent_id': torch.tensor(accent_id),
            'is_native': torch.tensor(float(is_native)),
            'transcription': row['transcription']
        }

def collate_fn(batch):
    # Implement custom collate function if needed
    return {key: [d[key] for d in batch] for key in batch[0]}

def extract_f0(self, waveform, sample_rate):
    # Convert to mono if stereo
    if waveform.ndim > 1:
        waveform = librosa.to_mono(waveform)
    
    # Extract pitch
    f0, _, _ = librosa.pyin(
        waveform, 
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        sr=sample_rate
    )
    
    # Replace NaN values (unvoiced regions) with 0
    f0 = np.nan_to_num(f0)
    return f0

def get_dataloader(csv_file, batch_size=32, num_workers=4):
    dataset = AccentDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)