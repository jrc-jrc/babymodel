import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def process_libritts(root_dir, output_dir):
    data = []
    os.makedirs(output_dir, exist_ok=True)
    
    for speaker_dir in tqdm(os.listdir(root_dir)):
        speaker_path = os.path.join(root_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for chapter_dir in os.listdir(speaker_path):
                chapter_path = os.path.join(speaker_path, chapter_dir)
                if os.path.isdir(chapter_path):
                    for file in os.listdir(chapter_path):
                        if file.endswith('.wav'):
                            # Process audio
                            audio_path = os.path.join(chapter_path, file)
                            y, sr = librosa.load(audio_path, sr=16000)
                            out_audio_path = os.path.join(output_dir, f"{speaker_dir}_{chapter_dir}_{file}")
                            sf.write(out_audio_path, y, sr)
                            
                            # Process corresponding text file
                            text_file = file.replace('.wav', '.normalized.txt')
                            text_path = os.path.join(chapter_path, text_file)
                            if os.path.exists(text_path):
                                with open(text_path, 'r') as f:
                                    text = f.read().strip()
                            else:
                                text = "Transcription not found"
                            
                            data.append({
                                'file_path': out_audio_path,
                                'speaker_id': speaker_dir,
                                'accent': 'American',
                                'transcription': text,
                                'gender': 'Unknown'
                            })
    return pd.DataFrame(data)

# Process LibriTTS dataset
libritts_data = process_libritts('data/audio/raw/libritts/LibriTTS/dev-clean', 'data/audio/processed/libritts')

# Save metadata
libritts_data.to_csv('libritts_metadata.csv', index=False)

# Create a smaller subset for the baby model
baby_data = libritts_data.sample(n=min(len(libritts_data), 1000))
baby_data.to_csv('libritts_baby_metadata.csv', index=False)

print(f"Total LibriTTS samples: {len(libritts_data)}")
print(f"Baby model samples: {len(baby_data)}")
print(libritts_data[['speaker_id', 'accent']].drop_duplicates())