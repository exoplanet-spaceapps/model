import os, json, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from .utils import seed_everything
def pr_auc(y, p): 
    try: return average_precision_score(y,p)
    except: return float("nan")
def roc_auc(y, p):
    try: return roc_auc_score(y,p)
    except: return float("nan")
def train(model, train_ds, val_ds, device="cpu", batch_size=64, lr=1e-3, weight_decay=1e-4, max_epochs=50, patience=7, workdir="."):
    seed_everything(42); os.makedirs(os.path.join(workdir,"artifacts"),exist_ok=True); os.makedirs(os.path.join(workdir,"reports"),exist_ok=True)
    tr=DataLoader(train_ds,batch_size=batch_size,shuffle=True); va=DataLoader(val_ds,batch_size=batch_size,shuffle=False)
    model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay); sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=0.5,patience=2)
    loss_fn=nn.BCEWithLogitsLoss(); best_ap=-1.0; wait=0; hist=[]
    for ep in range(1,max_epochs+1):
        model.train(); n=0; tl=0.0
        for G,L,Y in tr:
            G=torch.tensor(G,dtype=torch.float32,device=device); L=torch.tensor(L,dtype=torch.float32,device=device); Y=torch.tensor(Y,dtype=torch.float32,device=device)
            logit=model(G,L).squeeze(1); loss=loss_fn(logit,Y); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()*Y.shape[0]; n+=Y.shape[0]
        tl/=max(n,1)
        model.eval(); vs=[]; vy=[]
        with torch.no_grad():
            for G,L,Y in va:
                G=torch.tensor(G,dtype=torch.float32,device=device); L=torch.tensor(L,dtype=torch.float32,device=device); Y=torch.tensor(Y,dtype=torch.float32,device=device)
                logit=model(G,L).squeeze(1); vs.append(torch.sigmoid(logit).cpu().numpy()); vy.append(Y.cpu().numpy())
        vy=np.concatenate(vy) if vy else np.zeros(0); vp=np.concatenate(vs) if vs else np.zeros(0); ap=pr_auc(vy,vp); auc=roc_auc(vy,vp); sch.step(0.0 if np.isnan(ap) else ap)
        hist.append({"epoch":ep,"train_loss":float(tl),"val_ap":float(ap),"val_auc":float(auc)})
        if ap>best_ap: best_ap=ap; torch.save(model.state_dict(), os.path.join(workdir,"artifacts","cnn1d.pt")); wait=0
        else:
            wait+=1
            if wait>=patience: break
    with open(os.path.join(workdir,"reports","metrics_cnn.json"),"w",encoding="utf-8") as f: json.dump({"best_val_ap":float(best_ap),"history":hist},f,ensure_ascii=False,indent=2)
    return {"best_val_ap":float(best_ap)}
