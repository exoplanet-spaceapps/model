# app/trainers/cnn1d_trainer.py
from __future__ import annotations
import os, json, time
import numpy as np
from typing import Dict
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .utils import seed_everything

def pr_auc(y_true, y_prob):
    try: return average_precision_score(y_true, y_prob)
    except Exception: return float("nan")

def roc_auc(y_true, y_prob):
    try: return roc_auc_score(y_true, y_prob)
    except Exception: return float("nan")

def train(model: nn.Module, train_ds, val_ds, device:str="cpu",
          batch_size:int=64, lr:float=1e-3, weight_decay:float=1e-4,
          max_epochs:int=50, patience:int=7, workdir:str=".", seed:int=42) -> Dict:
    seed_everything(seed)
    os.makedirs(os.path.join(workdir, "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "reports"), exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2, verbose=False)
    loss_fn = nn.BCEWithLogitsLoss()

    best_ap = -1.0; wait=0
    best_path = os.path.join(workdir, "artifacts", "cnn1d.pt")
    history = []

    for epoch in range(1, max_epochs+1):
        model.train(); train_loss=0.0; n=0
        t0=time.perf_counter()
        for G,L,Y in train_loader:
            G=torch.tensor(G, dtype=torch.float32, device=device)
            L=torch.tensor(L, dtype=torch.float32, device=device)
            Y=torch.tensor(Y, dtype=torch.float32, device=device)
            logits = model(G,L).squeeze(1)
            loss = loss_fn(logits, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item()*Y.shape[0]; n+=Y.shape[0]
        train_loss/=max(n,1)

        # val
        model.eval(); vs=[]; vy=[]
        with torch.no_grad():
            for G,L,Y in val_loader:
                G=torch.tensor(G, dtype=torch.float32, device=device)
                L=torch.tensor(L, dtype=torch.float32, device=device)
                Y=torch.tensor(Y, dtype=torch.float32, device=device)
                logits=model(G,L).squeeze(1)
                vs.append(torch.sigmoid(logits).cpu().numpy()); vy.append(Y.cpu().numpy())
        vy=np.concatenate(vy) if vy else np.zeros(0); vp=np.concatenate(vs) if vs else np.zeros(0)
        ap=pr_auc(vy,vp); auc=roc_auc(vy,vp)
        sched.step(ap if not np.isnan(ap) else 0.0)
        history.append({"epoch":epoch,"train_loss":train_loss,"val_ap":float(ap),"val_auc":float(auc)})
        if ap>best_ap:
            best_ap=ap; torch.save(model.state_dict(), best_path); wait=0
        else:
            wait+=1
            if wait>=patience: break

    metrics={"best_val_ap":float(best_ap),"history":history}
    with open(os.path.join(workdir,"reports","metrics_cnn.json"),"w",encoding="utf-8") as f:
        json.dump(metrics,f,ensure_ascii=False,indent=2)
    return metrics
