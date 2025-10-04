import random, numpy as np, torch
def seed_everything(seed:int=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); 
    try: torch.cuda.manual_seed_all(seed)
    except: pass
