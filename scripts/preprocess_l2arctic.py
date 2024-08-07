import os
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm

def process_l2arctic(root_dir, output_dir):
    data = []
    os.makedirs(output_dir, exist_ok=True)
    
    accent_map = {
        'ABA': 'Arabic', 'SKA': 'Arabic', 'YBAA': 'Arabic', 'ZHAA': 'Arabic',
        'BWC': 'Mandarin', 'LXC': 'Mandarin', 'NCC': 'Mandarin', 'TXHC': 'Mandarin',
        'ASI': 'Hindi', 'RRBI': 'Hindi', 'SVBI': 'Hindi', 'TNI': 'Hindi',
        'HJK': 'Korean', 'HKK': 'Korean', 'YDCK': 'Korean', 'YKWK': 'Korean',
        'EBVS': 'Spanish', 'ERMS': 'Spanish', 'MBMPS': 'Spanish', 'NJS': 'Spanish',
        'HQTV': 'Vietnamese', 'PNV': 'Vietnamese', 'THV': 'Vietnamese', 'TLV': 'Vietnamese'
    }
    
    gender_map = {
        'ABA': 'M', 'SKA': 'F', 'YBAA': 'M', 'ZHAA': 'F',
        'BWC': 'M', 'LXC': 'F', 'NCC': 'F', 'TXHC': 'M',
        'ASI': 'M', 'RRBI': 'M', 'SVBI': 'F', 'TNI': 'F',
        'HJK': 'F', 'HKK': 'M', 'YDCK': 'F', 'YKWK': 'M',
        'EBVS': 'M', 'ERMS': 'M', 'MBMPS': 'F', 'NJS': 'F',
        'HQTV': 'M', 'PNV': 'F', 'THV': 'F', 'TLV': 'M'
    }
    
    for speaker_dir in tqdm(os.listdir(root_dir)):
        # DO this because there are nested folder with same names generated during unzip
        speaker_path = os.path.join(root_dir, speaker_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            wav_dir = os.path.join(speaker_path, 'wav')
            transcript_dir = os.path.join(speaker_path, 'transcript')
            
            for file in os.listdir(wav_dir):
                if file.endswith('.wav'):
                    # Process audio
                    audio_path = os.path.join(wav_dir, file)
                    y, sr = librosa.load(audio_path, sr=16000)
                    out_audio_path = os.path.join(output_dir, f"{speaker_dir}_{file}")
                    sf.write(out_audio_path, y, sr)
                    
                    # Process corresponding transcript file
                    text_file = file.replace('.wav', '.txt')
                    text_path = os.path.join(transcript_dir, text_file)
                    if os.path.exists(text_path):
                        with open(text_path, 'r') as f:
                            text = f.read().strip()
                    else:
                        text = "Transcription not found"
                    
                    data.append({
                        'file_path': out_audio_path,
                        'speaker_id': speaker_dir,
                        'accent': accent_map.get(speaker_dir, 'Unknown'),
                        'gender': gender_map.get(speaker_dir, 'Unknown'),
                        'transcription': text
                    })
    
    return pd.DataFrame(data)

# Process L2-Arctic dataset
l2arctic_data = process_l2arctic('data/audio/raw/l2arctic', 'data/audio/processed/l2arctic')

# Save metadata
l2arctic_data.to_csv('l2arctic_metadata.csv', index=False)

print(f"Total L2-Arctic samples: {len(l2arctic_data)}")
print(l2arctic_data['accent'].value_counts())
print(l2arctic_data['gender'].value_counts())