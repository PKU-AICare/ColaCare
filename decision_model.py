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
            # print(x.shape)
            x = layer(x)
        return x
    
    
class Merger(nn.Module):
    def __init__(self,model_emb_dim,text_emb_dim,output_dim,dropout=0.5):
        super(Merger, self).__init__()
        
        self.mtrans=TransformerEncoder(model_emb_dim+text_emb_dim)
        self.dropout=nn.Dropout(dropout)
        self.layer=nn.Linear(model_emb_dim+text_emb_dim,output_dim)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self,model_emb,text_emb):
        merge_embed=torch.concat((model_emb,text_emb),dim=1)
        trans_emb=self.mtrans(merge_embed)
        emb=self.dropout(trans_emb)
        emb=self.layer(emb)
        y=self.sigmoid(emb)
        return y
    
    
    
class Predictor(nn.Module):
    def __init__(self,input_dim,output_dim=1):
        super(Predictor, self).__init__()
        
        self.layer=nn.Linear(input_dim,output_dim)
        self.norm=nn.LayerNorm(input_dim)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,merge_emb):
        emb=self.norm(merge_emb)
        emb=self.layer(emb)
        y=self.sigmoid(emb)
        return y
    
    
    
class Decision(nn.Module):
    def __init__(self,model_embed_dim,text_embed_dim,merge_embed_dim,output_dim=1):
        super(Decision, self).__init__()     

        self.merge_layers= Merger(model_embed_dim, text_embed_dim, merge_embed_dim)
        self.predict_layers=Predictor(merge_embed_dim,output_dim)

    def forward(self,text_embedding,model_embedding):

        emb=self.merge_layers(model_embedding,text_embedding)
        output=self.predict_layers(emb)

        return output


if __name__== "__main__" :
    actor=Decision(model_embed_dim=64,text_embed_dim=64,merge_embed_dim=64)
    text_embeddings=[torch.rand(16,64),torch.rand(16,64)]
    model_embeddings=[torch.rand(16,64),torch.rand(16,64)]
    
    outputs,round=actor.forward(text_embedding=text_embeddings,model_embedding=model_embeddings)
    print(outputs[0])
    print(outputs[1])
    