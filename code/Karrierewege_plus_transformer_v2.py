import math
import re
from collections import defaultdict
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(tokens)

# Config
TOP_N_CLASSES = None
MAX_SEQ_LENGTH = 512
EMBED_DIM = 64
param_choices = {
    "num_layers": [1, 2],
    "num_heads": [2, 4],
    "batch_size": [32, 64],
    "learning_rate": [1e-3, 5e-4]
}
NUM_EPOCHS = 10

# Load dataset
print("Loading dataset...")
dataset = load_dataset("ElenaSenger/Karrierewege_plus")
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]
print(f"Dataset loaded: Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

print("\n--- Starting Job Title Clustering --- ")

# Collect unique labels and text
print("Collecting unique labels and representative text...")
label_to_representative_text = {}
unique_labels_in_train = set()
for rec in train_data:
    label = rec['preferredLabel_en']
    if label and label not in label_to_representative_text:
        title = rec.get('new_job_title_en_occ', '')
        desc = rec.get('new_job_description_en_cp', '')
        combined_text = f"{title} {desc}"
        processed_text = preprocess_text(combined_text)
        if processed_text:
            label_to_representative_text[label] = processed_text
            unique_labels_in_train.add(label)
print(f"Found {len(label_to_representative_text)} unique labels with text.")

# Prepare for embedding
cluster_target_labels = list(label_to_representative_text.keys())
texts_to_embed = [label_to_representative_text[lbl] for lbl in cluster_target_labels]

# Generate embeddings
print("Generating embeddings...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
embeddings = sentence_model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
print(f"Generated {embeddings.shape[0]} embeddings")

# Find optimal K
print("Finding optimal K using Elbow Method...")
k_range = range(10, 34, 1)
inertia = []

for k in k_range:
    print(f"Testing K={k}...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    inertia.append(kmeans.inertia_)

# Elbow point detection
delta_inertia = np.diff(inertia)
delta2_inertia = np.diff(delta_inertia)

if len(delta2_inertia) > 0:
    elbow_index = np.argmax(delta2_inertia) + 2
    best_k = k_range[elbow_index]
else:
    print("Not enough K values to find reliable elbow. Using default.")
    best_k = k_range[-1]

print(f"Optimal K = {best_k}")

# Plot elbow curve
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method For Optimal K')
plt.vlines(best_k, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='r')
plt.savefig('elbow_curve.png')

# Perform K-means clustering
print(f"Running K-means with K={best_k}...")
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_assignments = kmeans.fit_predict(embeddings)

# Create mapping
label_to_cluster_map = {label: cluster_id for label, cluster_id in zip(cluster_target_labels, cluster_assignments)}
print(f"Created map for {len(label_to_cluster_map)} labels to {best_k} clusters.")

# Inspect clusters
print("\nInspecting cluster contents:")
cluster_to_labels_map = defaultdict(list)
for label, cluster_id in label_to_cluster_map.items():
    cluster_to_labels_map[cluster_id].append(label)

for cluster_id in sorted(cluster_to_labels_map.keys()):
    labels_in_cluster = cluster_to_labels_map[cluster_id]
    print(f"Cluster {cluster_id} (Size: {len(labels_in_cluster)} labels):")
    for i, label in enumerate(labels_in_cluster[:10]):
        print(f"  - {label}")
    if len(labels_in_cluster) > 10:
        print("  - ... (and more)")

# Save cluster assignments
cluster_assignments_filename = "cluster_assignments.txt"
print(f"\nSaving cluster assignments to {cluster_assignments_filename}...")
with open(cluster_assignments_filename, 'w', encoding='utf-8') as f:
    for cluster_id in sorted(cluster_to_labels_map.keys()):
        labels_in_cluster = cluster_to_labels_map[cluster_id]
        f.write(f"--- Cluster {cluster_id} (Size: {len(labels_in_cluster)}) ---\\n")
        for label in sorted(labels_in_cluster):
            f.write(f"- {label}\\n")
        f.write("\\n")
print("Cluster assignments saved.")

# Process dataset for sequential modeling
def tokenize_text(text):
    tokens = re.findall(r"<sep>|[A-Za-z0-9]+", text.lower())
    return tokens

def make_sequence_samples(data, label_to_cluster_map):
    samples = []
    cluster_freq = defaultdict(int)
    current_resume = []
    for rec in data:
        if rec["experience_order"] == 0 and current_resume:
            if len(current_resume) > 1:
                job_info = [f"{job.get('new_job_title_en_occ', '')} {job.get('new_job_description_en_cp', '')}" for job in current_resume[:-1]]
                seq_text = " <sep> ".join(filter(None, job_info))
                tokens = tokenize_text(seq_text)
                tokens.insert(0, "<cls>")
                last_job_label = current_resume[-1].get("preferredLabel_en")
                target_cluster_id = label_to_cluster_map.get(last_job_label)
                if target_cluster_id is not None:
                    samples.append((tokens, target_cluster_id))
                    cluster_freq[target_cluster_id] += 1
            current_resume = []
        current_resume.append(rec)

    if current_resume and len(current_resume) > 1:
        job_info = [f"{job.get('new_job_title_en_occ', '')} {job.get('new_job_description_en_cp', '')}" for job in current_resume[:-1]]
        seq_text = " <sep> ".join(filter(None, job_info))
        tokens = tokenize_text(seq_text)
        tokens.insert(0, "<cls>")
        last_job_label = current_resume[-1].get("preferredLabel_en")
        target_cluster_id = label_to_cluster_map.get(last_job_label)
        if target_cluster_id is not None:
            samples.append((tokens, target_cluster_id))
            cluster_freq[target_cluster_id] += 1

    return samples, cluster_freq

print("Processing training data into sequences...")
train_samples, train_cluster_counts = make_sequence_samples(train_data, label_to_cluster_map)
print(f"Training sequences: {len(train_samples)}")

if TOP_N_CLASSES is not None:
    sorted_clusters = sorted(train_cluster_counts.items(), key=lambda x: x[1], reverse=True)
    top_clusters = {cluster_id for cluster_id, _ in sorted_clusters[:TOP_N_CLASSES]}
    train_samples = [(tokens, cluster_id) for tokens, cluster_id in train_samples if cluster_id in top_clusters]
    print(f"Filtered to top {TOP_N_CLASSES} clusters: {len(train_samples)} samples remaining.")
else:
    top_clusters = set(train_cluster_counts.keys())

print("Processing validation data...")
val_samples, val_cluster_counts = make_sequence_samples(val_data, label_to_cluster_map)
if TOP_N_CLASSES is not None:
    val_samples = [(tokens, cluster_id) for tokens, cluster_id in val_samples if cluster_id in top_clusters]
print(f"Validation sequences: {len(val_samples)}")

print("Processing test data...")
test_samples, test_cluster_counts = make_sequence_samples(test_data, label_to_cluster_map)
if TOP_N_CLASSES is not None:
    test_samples = [(tokens, cluster_id) for tokens, cluster_id in test_samples if cluster_id in top_clusters]
print(f"Test sequences: {len(test_samples)}")

# Build vocabulary
print("Building vocabulary...")
vocab = {"<PAD>": 0}
vocab["<CLS>"] = len(vocab)
vocab["<SEP>"] = len(vocab)
vocab["<UNK>"] = len(vocab)
for tokens, _ in train_samples:
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size} tokens")

# Map class labels to indices
num_classes = best_k
label_names = [f"Cluster {i}" for i in range(best_k)]
class_to_idx = {i: i for i in range(best_k)}

print(f"Number of target classes: {num_classes}")

# Calculate class weights
print("Calculating class weights...")
class_counts = torch.zeros(num_classes, dtype=torch.float)
for cluster_id, count in train_cluster_counts.items():
    if cluster_id in class_to_idx:
        class_counts[class_to_idx[cluster_id]] += count

total_samples = len(train_samples)
epsilon = 1e-6
class_weights = total_samples / (num_classes * (class_counts + epsilon))
class_weights = class_weights.to(device)
print(f"Class weights calculated")

def encode_tokens(tokens):
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(token_ids) < MAX_SEQ_LENGTH:
        token_ids += [vocab["<PAD>"]] * (MAX_SEQ_LENGTH - len(token_ids))
    else:
        token_ids = token_ids[:MAX_SEQ_LENGTH]
    return token_ids

print("Encoding sequences...")
X_train = [encode_tokens(tokens) for tokens, _ in train_samples]
y_train = [class_to_idx[cluster_id] for _, cluster_id in train_samples]
X_val = [encode_tokens(tokens) for tokens, _ in val_samples]
y_val = [class_to_idx[cluster_id] for _, cluster_id in val_samples]
X_test = [encode_tokens(tokens) for tokens, _ in test_samples]
y_test = [class_to_idx[cluster_id] for _, cluster_id in test_samples]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val = torch.tensor(X_val, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Dataset wrappers
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train, y_train)
val_dataset = SequenceDataset(X_val, y_val)
test_dataset = SequenceDataset(X_test, y_test)

# Define model
class CareerTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, num_classes, max_seq_len):
        super(CareerTransformer, self).__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab["<PAD>"])
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim*4, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        batch_size, seq_len = x.size()
        pos_indices = torch.arange(0, seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        x_emb = self.token_embed(x) + self.pos_embed(pos_indices)
        x_emb = x_emb.permute(1, 0, 2)
        pad_mask = (x == vocab["<PAD>"])
        enc_output = self.transformer_encoder(x_emb, src_key_padding_mask=pad_mask)
        cls_output = enc_output[0]
        logits = self.fc(cls_output)
        return logits

# Training functions
def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_X.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion, return_preds=False):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
            if return_preds:
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    if return_preds:
        return avg_loss, accuracy, all_preds, all_labels
    else:
        return avg_loss, accuracy

# Hyperparameter tuning
print("Starting hyperparameter tuning...")
best_val_loss = float('inf')
best_params = None
best_model_state = None
early_stopping_patience = 3

for num_layers in param_choices["num_layers"]:
    for num_heads in param_choices["num_heads"]:
        for batch_size in param_choices["batch_size"]:
            for lr in param_choices["learning_rate"]:
                print(f"\n--- Testing: layers={num_layers}, heads={num_heads}, batch={batch_size}, lr={lr:.4f} ---")
                model = CareerTransformer(vocab_size, EMBED_DIM, num_heads, num_layers, num_classes, MAX_SEQ_LENGTH).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                current_best_val_loss = float('inf')
                current_best_state = None
                epochs_no_improve = 0

                for epoch in range(1, NUM_EPOCHS + 1):
                    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
                    val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader, criterion, return_preds=True)
                    val_report = classification_report(val_labels, val_preds, labels=list(range(num_classes)), target_names=label_names, output_dict=True, zero_division=0)
                    current_val_f1 = val_report["macro avg"]["f1-score"]

                    print(f"  Epoch {epoch}/{NUM_EPOCHS} -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {current_val_f1:.4f}")

                    if val_loss < current_best_val_loss:
                        current_best_val_loss = val_loss
                        current_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                        epochs_no_improve = 0
                        print(f"    -> New best Val Loss: {current_best_val_loss:.4f}")
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= early_stopping_patience:
                        print(f"    -> Early stopping after epoch {epoch}")
                        break

                print(f"Best Val Loss for this config: {current_best_val_loss:.4f}")
                if current_best_val_loss < best_val_loss:
                    print(f"  *** New overall best parameters! ***")
                    best_val_loss = current_best_val_loss
                    best_params = (num_layers, num_heads, batch_size, lr)
                    best_model_state = current_best_state

# Use best hyperparameters
best_num_layers, best_num_heads, best_batch_size, best_lr = best_params
print(f"\nBest hyperparameters: layers={best_num_layers}, heads={best_num_heads}, batch_size={best_batch_size}, lr={best_lr:.4f}")
print(f"Lowest Validation Loss: {best_val_loss:.4f}")

# Final evaluation
print("\nEvaluating best model on test set...")
best_model = CareerTransformer(vocab_size, EMBED_DIM, best_num_heads, best_num_layers, num_classes, MAX_SEQ_LENGTH).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
best_model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)
test_loss, test_acc, test_preds, test_labels = evaluate_model(best_model, test_loader, criterion, return_preds=True)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
print("Classification Report:")
print(classification_report(test_labels, test_preds, labels=list(range(num_classes)), target_names=label_names, digits=4, zero_division=0))
cm = confusion_matrix(test_labels, test_preds, labels=list(range(num_classes)))
cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
print("Confusion Matrix:")
print(cm_df)

# Save model
print("\nSaving the best model...")
save_path = "best_career_transformer_model_clustered.pth"
model_config = {
    'vocab_size': vocab_size,
    'embed_dim': EMBED_DIM,
    'num_heads': best_num_heads,
    'num_layers': best_num_layers,
    'num_classes': num_classes,
    'max_seq_len': MAX_SEQ_LENGTH,
    'vocab': vocab,
    'label_names': label_names,
    'is_clustered': True,
    'best_k': best_k
}
torch.save({
    'model_state_dict': best_model_state,
    'config': model_config
}, save_path)
print(f"Model saved to {save_path}")

# Prediction helper
def predict_next_job_cluster(history, model, config, device):
    print(f"\nInput Career History: {history}")

    vocab = config['vocab']
    label_names = config['label_names']
    max_seq_len = config['max_seq_len']
    num_clusters = config['num_classes']

    seq_text = " <sep> ".join(history)
    tokens = tokenize_text(seq_text)
    tokens.insert(0, "<cls>")

    encoded_sequence = []
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    if len(token_ids) < max_seq_len:
        encoded_sequence = token_ids + [vocab["<PAD>"]] * (max_seq_len - len(token_ids))
    else:
        encoded_sequence = token_ids[:max_seq_len]

    input_tensor = torch.tensor([encoded_sequence], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_k = 5
    top_probs, top_indices = torch.topk(probabilities, top_k)

    print(f"Top {top_k} Predicted Next Job Clusters:")
    for i in range(top_k):
        pred_idx = top_indices[i].item()
        if 0 <= pred_idx < num_clusters:
            pred_label = label_names[pred_idx]
            pred_prob = top_probs[i].item()
            print(f"  {i+1}. {pred_label} (Confidence: {pred_prob:.4f})")
        else:
            print(f"  {i+1}. Error: Predicted index {pred_idx} out of bounds (Num clusters: {num_clusters})")
    print("-" * 20)

# Example predictions
print("\n--- Example Predictions ---")

checkpoint = torch.load(save_path, map_location=device)
saved_config = checkpoint['config']

example_1 = ["research assistant", "doctoral researcher", "postdoctoral researcher"]
predict_next_job_cluster(example_1, best_model, saved_config, device)

example_2 = ["construction laborer", "metal fabricator"]
predict_next_job_cluster(example_2, best_model, saved_config, device)

example_3 = ["kitchen assistant", "warehouse worker"]
predict_next_job_cluster(example_3, best_model, saved_config, device)

example_4 = ["administrative assistant", "logistics coordinator"]
predict_next_job_cluster(example_4, best_model, saved_config, device)

example_5 = ["retail sales associate", "retail support staff", "inventory coordinator"]
predict_next_job_cluster(example_5, best_model, saved_config, device)
