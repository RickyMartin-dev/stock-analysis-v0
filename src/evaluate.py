# import torch
# import torch.nn as nn
# import numpy as np
# import pandas as pd
# from sklearn.metrics import accuracy_score, roc_auc_score

# def evaluate_metrics(model, dl):
#     model.eval()
#     probs, targs = [], []
#     with torch.no_grad():
#         for xb, yb in dl:
#             logits = model(xb)
#             p = torch.sigmoid(logits).cpu().numpy()
#             probs.append(p)
#             targs.append(yb.cpu().numpy())
#     probs = np.concatenate(probs)
#     targs = np.concatenate(targs)
#     preds = (probs >= 0.5).astype(int)
#     acc = accuracy_score(targs, preds)
#     try:
#         auc = roc_auc_score(targs, probs)
#     except Exception:
#         auc = float("nan")
#     return acc, auc, probs, targs, preds