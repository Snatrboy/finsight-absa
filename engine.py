import json, os, gc, threading
from pathlib import Path
import numpy as np
import joblib
from huggingface_hub import snapshot_download
import re
SLANG = {
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "nggak": "tidak", "ngga": "tidak", "tdk": "tidak",
    "tak": "tidak", "enggak": "tidak", "gabisa": "tidak bisa", "gbs": "tidak bisa", "gamau": "tidak mau",
    "gada": "tidak ada", "udh": "sudah", "udah": "sudah", "sdh": "sudah", "dah": "sudah",
    "blm": "belum", "belom": "belum", "yg": "yang", "dgn": "dengan", "dg": "dengan", "utk": "untuk",
    "krn": "karena", "karna": "karena", "trf": "transfer", "tf": "transfer", "trx": "transaksi",
    "sy": "saya", "aq": "aku", "bgt": "banget", "bngt": "banget", "knp": "kenapa", "tp": "tapi",
    "tpi": "tapi", "jg": "juga", "trs": "terus", "msh": "masih", "dr": "dari", "bs": "bisa",
    "bsa": "bisa", "jd": "jadi", "sm": "sama", "org": "orang", "hrs": "harus", "lg": "lagi",
    "lgi": "lagi", "dpt": "dapat", "dlm": "dalam", "pdhl": "padahal", "bkn": "bukan", "kyk": "seperti",
}

TOKEN_WORD = re.compile(r"[a-z]+\d*|\d+[a-z]*")

EMO_POS = re.compile("[\U0001F44D\U0001F44F\U0001F44C\U0001F60A\U0001F60D\U0001F600-\U0001F604\U0001F970\U0001F618\u2764\U0001F499\U0001F49A\U0001F64F\U0001F44C\U0001F525\u2705\U0001F929\U0001F973]")

EMO_NEG = re.compile("[\U0001F621\U0001F620\U0001F92C\U0001F44E\U0001F62D\U0001F622\U0001F624\U0001F61E\U0001F614\U0001F629\U0001F62B\U0001F612\U0001F92E\U0001F4A2]")

def normalize_model_v2(text):
    """Teks untuk model: huruf kecil, URL dibuang, emoji sentimen menjadi kata (emo_pos/emo_neg), tanda baca dibuang, slang dibakukan."""
    t = str(text)
    t = EMO_POS.sub(" emo_pos ", t); t = EMO_NEG.sub(" emo_neg ", t)
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
    t = t.replace("emo_pos", " EMOPOS ").replace("emo_neg", " EMONEG ")
    t = re.sub(r"[^\x00-\x7f]+", " ", t)
    t = re.sub(r"([a-z])\1{2,}", r"\1\1", t)
    out = []
    for w in TOKEN_WORD.findall(t.replace("EMOPOS", "emopos").replace("EMONEG", "emoneg")):
        w = {"emopos": "emo_pos", "emoneg": "emo_neg"}.get(w, SLANG.get(w, w))
        out.extend(w.split())
    return " ".join(out)
LOCK=threading.RLock()
ACTIVE=None

def predict(m,texts,connection,token):
    global ACTIVE
    with LOCK:
        key=(connection['repo_id'],connection['revision'],m['artifact_id'])
        if ACTIVE is None or ACTIVE[0]!=key:
            ACTIVE=None; gc.collect()
            local=os.environ.get('FINSIGHT_MODEL_DIR')
            root=Path(local) if local else Path(snapshot_download(repo_id=key[0],revision=key[1],token=token,allow_patterns=[key[2]+'/*']))
            folder=root/key[2]
            if m['family'] in ('dt','lr','majority'):
                obj=joblib.load(folder/'model.joblib'); aux=None
            else:
                from neural import load_neural
                obj,aux=load_neural(m,folder)
            ACTIVE=(key,obj,aux)
        _,obj,aux=ACTIVE
        texts=[normalize_model_v2(t) for t in texts]
        if m['family'] in ('dt','lr','majority'):
            out=np.zeros((len(texts),6,4))
            for j,c in enumerate(m['labels']):
                kind,e=obj[c]
                if kind=='const': out[:,j,e]=1
                else: out[:,j][:,e.classes_]=e.predict_proba(texts)
        else:
            from neural import infer
            out=infer(m,obj,aux,texts)
        if out.shape!=(len(texts),6,4) or not np.isfinite(out).all() or not np.allclose(out.sum(-1),1,atol=1e-5): raise ValueError('Output model tidak valid')
        return out
