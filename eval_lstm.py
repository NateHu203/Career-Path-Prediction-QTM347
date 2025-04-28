import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from huggingface_hub import login
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Set device
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Login to Hugging Face
login("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")

def load_from_csv():
    train_df_raw = pd.read_csv('train_df.csv')
    validation_df_raw = pd.read_csv('val_df.csv')
    test_df_raw = pd.read_csv('test_df.csv')
    return train_df_raw, validation_df_raw, test_df_raw

def load_and_save_data():
    ds = load_dataset("ElenaSenger/Karrierewege_plus")
    train_df_raw = pd.DataFrame(ds['train'])
    validation_df_raw = pd.DataFrame(ds['validation'])
    test_df_raw = pd.DataFrame(ds['test'])
    
    # Save as CSVs
    train_df_raw.to_csv('train_df.csv', index=False)
    validation_df_raw.to_csv('val_df.csv', index=False)
    test_df_raw.to_csv('test_df.csv', index=False)
    
    return train_df_raw, validation_df_raw, test_df_raw

def create_job_vocab(train_df_raw, validation_df_raw, test_df_raw):
    job_titles = set(train_df_raw['new_job_title_en_cp']).union(
        validation_df_raw['new_job_title_en_cp'], 
        test_df_raw['new_job_title_en_cp']
    )
    
    job_vocab = {}
    for i, job_title in enumerate(sorted(job_titles), start=1):
        job_vocab[job_title] = i
    
    idx_to_job = {}
    for title, idx in job_vocab.items():
        idx_to_job[idx] = title
    
    # Special tokens
    job_vocab['<PAD>'] = 0 
    idx_to_job[0] = '<PAD>'
    unknown_idx = len(job_vocab)
    job_vocab['<UNK>'] = unknown_idx
    idx_to_job[unknown_idx] = '<UNK>'
    
    return job_vocab, idx_to_job, unknown_idx

def match_titles_to_descriptions(df, job_vocab, field_name="new_job_description_en_cp"):
    # Match job titles to descriptions
    job_desc_dict = {}
    for _, row in df.iterrows():
        title = row['new_job_title_en_cp']
        job_desc = row[field_name]

        if title not in job_desc_dict and pd.notna(job_desc): 
            job_desc_dict[title] = job_desc

    # Sorted list of job titles based on indices
    job_titles_sorted = [title for title, idx in sorted(job_vocab.items(), key=lambda x: x[1]) if title not in ['<PAD>', '<UNK>']]

    # Descriptions corresponding to the sorted job titles
    descriptions = []
    for title in job_titles_sorted:
        descriptions.append(job_desc_dict.get(title, ""))

    return descriptions

def tfidf_feats(df, job_vocab, field_name="new_job_description_en_cp", max_features=100):
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Match
    descriptions = match_titles_to_descriptions(df, job_vocab, field_name)
    
    # Fit and transform
    vectorizer = TfidfVectorizer(max_features=max_features)
    desc_features = vectorizer.fit_transform(descriptions).toarray()
    
    # Add one additional row for padding
    padding_features = np.zeros((1, desc_features.shape[1]))
    unknown_features = np.mean(desc_features, axis=0, keepdims=True)
    final_features = np.vstack([padding_features, desc_features, unknown_features])
    
    return final_features

class CareerLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        feat_dim: int,
        hid_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        lstm_input_dim = emb_dim + feat_dim
        self.lstm = nn.LSTM(
            lstm_input_dim,
            hid_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hid_dim * 2 if bidirectional else hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, lengths, job_feats):
        emb = self.embedding(seqs)  # (B, L, emb_dim)
        feats = job_feats[seqs]     # (B, L, feat_dim)
        x = torch.cat([emb, feats], dim=2)  # (B, L, emb+feat)
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        
        # Concatenate forward and backward hidden states if bidirectional
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        hidden = self.dropout(hidden)
        return self.fc(hidden)  # (B, vocab_size)

def predict_next_career(
    model: nn.Module,
    title_sequence: list[str],
    job_vocab: dict[str, int],
    idx_to_job: dict[int, str],
    job_feats: torch.Tensor,
    device: torch.device,
    top_k: int = 5
):
    model.eval()
    idxs = [job_vocab.get(t, job_vocab['<UNK>']) for t in title_sequence]
    seq = torch.LongTensor([idxs]).to(device)
    length = torch.LongTensor([len(idxs)]).to(device)

    with torch.no_grad():
        logits = model(seq, length, job_feats)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = probs.topk(top_k)
        predictions = [
            (idx_to_job.get(int(idx), "<UNK>"), float(prob))
            for idx, prob in zip(top_indices, top_probs)
        ]
        return predictions

def main():
    # Try to load data from CSV first, if files don't exist, download from HF
    try:
        train_df_raw, validation_df_raw, test_df_raw = load_from_csv()
    except:
        train_df_raw, validation_df_raw, test_df_raw = load_and_save_data()
    
    # Create job vocabulary
    job_vocab, idx_to_job, unknown_idx = create_job_vocab(train_df_raw, validation_df_raw, test_df_raw)
    
    # Feature extraction 
    full_df = pd.concat([train_df_raw, validation_df_raw, test_df_raw])
    job_features = tfidf_feats(full_df, job_vocab, max_features=100)
    job_features_tensor = torch.FloatTensor(job_features).to(device)
    
    # Define model parameters - must match the ones used for training
    vocab_size = len(job_vocab)
    feature_dim = job_features_tensor.size(1)
    embedding_dim = 128
    hidden_dim = 256
    output_dim = vocab_size
    
    # Create model
    model = CareerLSTM(vocab_size, embedding_dim, feature_dim, hidden_dim, output_dim).to(device)
    
    # Load trained model
    try:
        model.load_state_dict(torch.load("career_lstm_best.pt"))
        print("Model loaded successfully!")
    except:
        print("Error loading model. Make sure 'career_lstm_best.pt' exists.")
        return
    
    # Example career paths to predict from
    example_career_paths = [
        ["Office Support Specialist", "Financial Operations Coordinator", "Administrative Professional"],
        ["Cashier", "Retail Sales Associate", "Store Manager"],
        ["Software Developer", "Senior Software Engineer", "Team Lead"],
        ["Nurse", "Senior Nurse", "Nursing Supervisor"]
    ]
    
    # Make predictions
    print("\nPredicting next career steps:")
    print("-" * 50)
    
    for career_path in example_career_paths:
        print(f"\nCareer Path: {' -> '.join(career_path)}")
        predictions = predict_next_career(
            model,
            career_path,
            job_vocab,
            idx_to_job,
            job_features_tensor,
            device,
            top_k=5
        )
        
        print("Top 5 Predicted Next Jobs:")
        for i, (title, prob) in enumerate(predictions, 1):
            print(f"{i}. {title} (Confidence: {prob:.4f})")
        
        print("-" * 50)

if __name__ == "__main__":
    main() 