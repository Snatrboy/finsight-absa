import json
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer
class BiLSTMABSA(nn.Module):
    def __init__(self, vocab_size, dim, hidden, dropout):
        super().__init__()
        self.emb=nn.Embedding(vocab_size,dim,padding_idx=0)
        self.drop=nn.Dropout(dropout)
        self.lstm=nn.LSTM(dim,hidden,batch_first=True,bidirectional=True)
        self.out=nn.Linear(4*hidden,24)
    def forward(self,x,lens):
        e=self.drop(self.emb(x))
        h,_=self.lstm(nn.utils.rnn.pack_padded_sequence(e,lens.cpu(),batch_first=True,enforce_sorted=False))
        h,_=nn.utils.rnn.pad_packed_sequence(h,batch_first=True,total_length=x.size(1))
        m=(x!=0).unsqueeze(-1)
        return self.out(self.drop(torch.cat([h.masked_fill(~m,-1e9).max(1).values,(h*m).sum(1)/m.sum(1).clamp(min=1)],1))).view(-1,6,4)
class BertABSA(nn.Module):
    def __init__(self,config,pool):
        super().__init__(); self.bert=AutoModel.from_config(config); self.pool=pool
        self.drop=nn.Dropout(.1); self.head=nn.Linear(config.hidden_size,24)
    def forward(self,ids,att):
        h=self.bert(input_ids=ids,attention_mask=att).last_hidden_state
        v=h[:,0] if self.pool=='cls' else (h*att.unsqueeze(-1)).sum(1)/att.sum(1,keepdim=True).clamp(min=1)
        return self.head(self.drop(v)).view(-1,6,4)

def load_neural(m,p):
    state=torch.load(p/'model.pt',map_location='cpu',weights_only=True)
    if m['family']=='lstm':
        vocab=json.loads((p/'vocab.json').read_text()); size,dim=state['emb.weight'].shape
        assert size==len(vocab)
        model=BiLSTMABSA(size,dim,m['config']['hidden'],m['config']['dropout']); aux={w:i for i,w in enumerate(vocab)}
    else:
        model=BertABSA(AutoConfig.from_pretrained(p/'encoder_config',local_files_only=True),m['pool']); aux=AutoTokenizer.from_pretrained(p/'tokenizer',local_files_only=True)
    model.load_state_dict(state,strict=True); del state
    return model.eval(),aux
@torch.inference_mode()
def infer(m,model,aux,texts):
    if m['family']=='lstm':
        ids=[[aux.get(w,1) for w in t.split()][:m['maxlen']] or [1] for t in texts]
        x=torch.zeros(len(ids),m['maxlen'],dtype=torch.long)
        for i,row in enumerate(ids): x[i,:len(row)]=torch.tensor(row)
        args=(x,torch.tensor([len(r) for r in ids]))
    else:
        b=aux(texts,padding=True,truncation=True,max_length=m['maxlen'],return_tensors='pt'); args=(b.input_ids,b.attention_mask)
    return model(*args).softmax(-1).cpu().numpy()
