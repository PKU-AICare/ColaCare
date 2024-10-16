import ipdb
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, emb_dim, num_heads=4, hidden_dim=256):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=num_heads, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, x, mask=None):
        x_res = x
        x = self.ln1(x)
        x, _ = self.mha(x, x, x)
        x += x_res
        x_res = x
        x = self.ln2(x_res)
        x = self.ffn(x)
        x += x_res
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, num_heads=4, hidden_dim=256, num_layers=2):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(emb_dim, num_heads, hidden_dim)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class Merger(nn.Module):
    def __init__(self, ehr_embed_dim, ehr_num, text_emb_dim, hidden_dim):
        super(Merger, self).__init__()
        self.layer = nn.Linear(ehr_embed_dim + text_emb_dim, hidden_dim)
        self.act = nn.GELU() 
    
    def forward(self, ehr_embs, text_emb):
        emb = torch.cat(ehr_embs + [text_emb], dim=1)
        emb = self.layer(emb)
        return emb


class Predictor(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(Predictor, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, emb):
        emb = self.layer(emb)
        y = self.sigmoid(emb)
        return y


class Fusion(nn.Module):
    def __init__(self, ehr_embed_dim, ehr_num, text_embed_dim, merge_embed_dim, output_dim=1):
        super(Fusion, self).__init__()     
        self.merge_layers = Merger(ehr_embed_dim, ehr_num, text_embed_dim, merge_embed_dim)
        self.predict_layers = Predictor(merge_embed_dim, output_dim)

    def forward(self, ehr_embeddings, text_embedding, *args, **kwargs):
        emb = self.merge_layers(ehr_embeddings, text_embedding)
        output = self.predict_layers(emb)

        return output.squeeze(-1)