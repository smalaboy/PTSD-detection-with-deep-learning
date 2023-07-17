
"""# Moctar, imports"""

from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import os
import glob
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import KFold, train_test_split
import random
import warnings
warnings.filterwarnings("ignore")
from transformers import logging
logging.set_verbosity_error()

"""# Moctar, Presets"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""# Moctar, classes"""

class PTSD_Text_Dataset(Dataset):
    def __init__(self, paths, labels, tokenizer):
        self.paths = paths
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        with open(path, "r") as f:
            text = f.read().lower()
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )
        tokens["input_ids"] = tokens["input_ids"].squeeze()
        tokens["attention_mask"] = tokens["attention_mask"].squeeze()
        tokens["token_type_ids"] = tokens["token_type_ids"].squeeze()

        return tokens, label


class Model(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 2)

    def forward(self, tokens):
        embeddings = self.bert_model(**tokens).last_hidden_state
        embeddings = embeddings.mean(dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.fc(embeddings)
        return logits

"""# Moctar, train for single train-val-test split"""

torch.cuda.empty_cache()

train_paths_ptsd = glob.glob("train/PTSD/**/*.txt", recursive=True)
train_paths_no_ptsd = glob.glob("train/NO PTSD/**/*.txt", recursive=True)

validation_paths_ptsd = glob.glob("validation/PTSD/**/*.txt", recursive=True)
validation_paths_no_ptsd = glob.glob("validation/NO PTSD/**/*.txt", recursive=True)

test_paths_ptsd = glob.glob("test/PTSD/**/*.txt", recursive=True)
test_paths_no_ptsd = glob.glob("test/NO PTSD/**/*.txt", recursive=True)

print(
    f"\tN_train_PTSD: {len(train_paths_ptsd)}",
    f"\tN_train_NO_PTSD: {len(train_paths_no_ptsd)}",
)
print(
    f"\tN_validation_PTSD: {len(validation_paths_ptsd)}",
    f"\tN_validation_NO_PTSD: {len(validation_paths_no_ptsd)}",
)
print(
    f"\tN_test_PTSD: {len(test_paths_ptsd)}",
    f"\tN_test_NO_PTSD: {len(test_paths_no_ptsd)}",
)

train_paths = train_paths_ptsd + train_paths_no_ptsd
train_labels = [1] * len(train_paths_ptsd) + [0] * len(train_paths_no_ptsd)

validation_paths = validation_paths_ptsd + validation_paths_no_ptsd
validation_labels = [1] * len(validation_paths_ptsd) + [0] * len(validation_paths_no_ptsd)

test_paths = test_paths_ptsd + test_paths_no_ptsd
test_labels = [1] * len(test_paths_ptsd) + [0] * len(test_paths_no_ptsd)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = PTSD_Text_Dataset(train_paths, train_labels, tokenizer)
val_dataset = PTSD_Text_Dataset(validation_paths, validation_labels, tokenizer)
test_dataset = PTSD_Text_Dataset(test_paths, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

criterion = nn.CrossEntropyLoss().to(device)
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(bert_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, eps = 1e-08)

best_val_f1 = 0.0
for epoch in range(1, 6):
    print(f"\tEpoch: {epoch}/5")

    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        tokens, labels = batch
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"\t\t{i+1}/{len(train_loader)} Train Loss: {loss.item()}")

    model.eval()
    eval_pred_list, eval_label_list = [], []
    for batch in val_loader:
        tokens, labels = batch
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(tokens)

        preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()

        eval_pred_list.extend(preds.tolist())
        eval_label_list.extend(labels_np.tolist())

    val_f1 = f1_score(eval_label_list, eval_pred_list)
    print(f"\t\tValidation F1: {val_f1}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "weights_best_val_f1.pt")

"""# Moctar, test for single train-val-test split"""

model.eval()
print(model.load_state_dict(torch.load("weights_best_val_f1.pt")))

test_pred_list = []
test_label_list = []
for batch in test_loader:
    tokens, labels = batch
    tokens = tokens.to(device)
    labels = labels.to(device)
    logits = model(tokens)

    preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()

    test_pred_list.extend(preds.tolist())
    test_label_list.extend(labels_np.tolist())

test_f1 = f1_score(test_label_list, test_pred_list)
test_acc = accuracy_score(test_label_list, test_pred_list)
test_prec = precision_score(test_label_list, test_pred_list)
test_rec = recall_score(test_label_list, test_pred_list)

print(
    f"\tTesting Results For Single Split\n"
    f"\t\tAcc: {test_acc}\n"
    f"\t\tPrec: {test_prec}\n"
    f"\t\tRec: {test_rec}\n"
    f"\t\tF1: {test_f1}\n"
)

"""# Moctar, 3-Fold CV
## Important, change n_fold at line 10 to the one of [0,1,2]
"""

def read_list_from_file(path):
    l = []
    with open(path, "r") as f:
        return list(map(lambda p: p.replace("\n", ""), f.readlines()))



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.empty_cache()

# IMPORTANT
# Change n_fold to which split you are working on, one of [0,1,2]
n_fold = 2

train_paths = read_list_from_file(f"text_split{n_fold}/train_paths.txt")
test_paths = read_list_from_file(f"text_split{n_fold}/test_paths.txt")

train_paths_ptsd = list(filter(lambda p: p.split("/")[0] == "PTSD", train_paths))
train_paths_no_ptsd = list(filter(lambda p: p.split("/")[0] == "NO PTSD", train_paths))
test_paths_ptsd = list(filter(lambda p: p.split("/")[0] == "PTSD", test_paths))
test_paths_no_ptsd = list(filter(lambda p: p.split("/")[0] == "NO PTSD", test_paths))

print(f"FOLD: {n_fold+1}/3")
print(
    f"\tN_train_PTSD: {len(train_paths_ptsd)}",
    f"\tN_train_NO_PTSD: {len(train_paths_no_ptsd)}",
)
print(
    f"\tN_test_PTSD: {len(test_paths_ptsd)}",
    f"\tN_test_NO_PTSD: {len(test_paths_no_ptsd)}",
)

train_paths  = train_paths_ptsd + train_paths_no_ptsd
train_labels = [1] * len(train_paths_ptsd) + [0] * len(train_paths_no_ptsd)

test_paths   = test_paths_ptsd + test_paths_no_ptsd
test_labels  = [1] * len(test_paths_ptsd) + [0] * len(test_paths_no_ptsd)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

train_dataset = PTSD_Text_Dataset(train_paths, train_labels, tokenizer)
test_dataset = PTSD_Text_Dataset(test_paths, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

criterion = nn.CrossEntropyLoss().to(device)
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
model = Model(bert_model).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr = 5e-5, eps = 1e-08)

for epoch in range(1, 6):
    print(f"\tEpoch: {epoch}/5")

    model.train()
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        tokens, labels = batch
        tokens = tokens.to(device)
        labels = labels.to(device)
        logits = model(tokens)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        print(f"\t\t{i+1}/{len(train_loader)} Train Loss: {loss.item()}")

torch.save(model.state_dict(), f"weights_split{n_fold}.pt")

"""# Moctar, test for nth fold"""

model.eval()
print(model.load_state_dict(torch.load(f"weights_split{n_fold}.pt")))

test_pred_list = []
test_label_list = []
for batch in test_loader:
    tokens, labels = batch
    tokens = tokens.to(device)
    labels = labels.to(device)
    logits = model(tokens)

    preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()

    test_pred_list.extend(preds.tolist())
    test_label_list.extend(labels_np.tolist())

test_f1 = f1_score(test_label_list, test_pred_list)
test_acc = accuracy_score(test_label_list, test_pred_list)
test_prec = precision_score(test_label_list, test_pred_list)
test_rec = recall_score(test_label_list, test_pred_list)

print(
    f"\tTesting Results For fold = {n_fold}\n"
    f"\t\tAcc: {test_acc}\n"
    f"\t\tPrec: {test_prec}\n"
    f"\t\tRec: {test_rec}\n"
    f"\t\tF1: {test_f1}\n"
)
# 
# Testing Results For fold = 0
# 		Acc: 0.9858490566037735
# 		Prec: 0.98
# 		Rec: 0.98989898989899
# 		F1: 0.9849246231155778
# 
# Testing Results For fold = 1
# 		Acc: 0.9715639810426541
# 		Prec: 0.956140350877193
# 		Rec: 0.990909090909091
# 		F1: 0.9732142857142858
# 
# Testing Results For fold = 2
# 		Acc: 0.9715639810426541
# 		Prec: 0.9722222222222222
# 		Rec: 0.9722222222222222
# 		F1: 0.9722222222222222
# 
# (0.9858490566037735 + 0.9715639810426541 + 0.9715639810426541) / 3
# 
# (0.98 + 0.956140350877193 + 0.9722222222222222) / 3
# 
# (0.98989898989899 + 0.990909090909091 + 0.9722222222222222) / 3
# 
# (0.9849246231155778 + 0.9732142857142858 + 0.9722222222222222) / 3
