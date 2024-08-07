# pronunciation_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PronunciationEncoder(nn.Module):
    def __init__(self, num_accents, d_model=512, nhead=8, num_layers=4, dropout=0.3):
        super(PronunciationEncoder, self).__init__()
        self.accent_embedding = nn.Embedding(num_accents, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout),
            num_layers
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_predictions, accent_id):
        # unsqueeze(1) adds a new dimension at index 1
        # expand(-1,  text_predictions.size(1), -1) only change the dimension on index 1 to the text_prediction sequence length
        # as the model normally ouput matrix in the form of (batch_size, sequence_length, feature_dim)
        accent_emb = self.accent_embedding(accent_id).unsqueeze(1).expand(-1, text_predictions.size(1), -1)
        x = torch.cat([text_predictions, accent_emb], dim=-1)
        x = self.dropout(x)
        return self.transformer(x.transpose(0, 1)).transpose(0, 1)