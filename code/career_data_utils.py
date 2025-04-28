import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

def transform_career_data_without_split(kartierung_df, batch_size=64):
    """
    Transform the raw career data into a format suitable for LSTM training.
    No train/test split is performed.
    
    Args:
        kartierung_df: DataFrame containing the career sequence data
        batch_size: Batch size for DataLoader
    
    Returns:
        data_loader: DataLoader for the data
        job_vocab: Dictionary mapping job titles to indices
        idx_to_job: Dictionary mapping indices to job titles
    """
    print("Grouping career sequences by person ID...")
    
    # Group data by person ID
    career_sequences = []
    job_titles = set()
    
    # Dictionary to store career sequences by person
    person_careers = defaultdict(list)
    
    # First pass: collect all jobs for each person
    for _, row in kartierung_df.iterrows():
        person_id = row['_id']
        job_title = row['new_job_title_en_occ']
        sequence = row['experience_order']
        
        person_careers[person_id].append((sequence, job_title))
        job_titles.add(job_title)
    
    # Second pass: sort by sequence and create career paths
    for person_id, jobs in person_careers.items():
        # Sort by sequence number
        sorted_jobs = [job for _, job in sorted(jobs, key=lambda x: x[0])]
        
        # Only include if person has at least 2 jobs (for prediction)
        if len(sorted_jobs) >= 2:
            career_sequences.append(sorted_jobs)
    
    print(f"Found {len(career_sequences)} valid career sequences")
    print(f"Found {len(job_titles)} unique job titles")
    
    # Create job vocabulary (reserve 0 for padding)
    job_vocab = {job: idx for idx, job in enumerate(sorted(job_titles), start=1)}
    idx_to_job = {idx: job for job, idx in job_vocab.items()}
    
    # Create dataset
    dataset = SimpleCareerDataset(career_sequences, job_vocab)
    
    # Create DataLoader with custom collate function for padding
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,  # Set to False for test data
        collate_fn=collate_pad_sequences
    )
    
    return data_loader, job_vocab, idx_to_job

class SimpleCareerDataset(torch.utils.data.Dataset):
    """Dataset for career sequence data."""
    
    def __init__(self, career_sequences, job_vocab):
        self.sequences = career_sequences
        self.job_vocab = job_vocab
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Convert job titles to indices
        job_indices = [self.job_vocab.get(job, len(self.job_vocab)) for job in sequence]
        
        # Input is all jobs except the last one
        input_seq = job_indices[:-1]
        
        # Target is the last job
        target = job_indices[-1]
        
        return {
            'input_seq': input_seq,
            'target': target,
            'length': len(input_seq)
        }

def collate_pad_sequences(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the maximum length in the batch.
    """
    # Find maximum sequence length in this batch
    max_length = max([item['length'] for item in batch])
    
    # Prepare batch tensors
    input_seqs = []
    targets = []
    lengths = []
    
    for item in batch:
        # Pad sequence to max_length
        padded = item['input_seq'] + [0] * (max_length - len(item['input_seq']))
        input_seqs.append(padded)
        targets.append(item['target'])
        lengths.append(item['length'])
    
    return {
        'input_seqs': torch.tensor(input_seqs, dtype=torch.long),
        'targets': torch.tensor(targets, dtype=torch.long),
        'lengths': torch.tensor(lengths, dtype=torch.long)
    }

def process_job_descriptions(kartierung_df, job_vocab):
    """
    Process job descriptions to create feature vectors for each job.
    This uses a simple approach with TF-IDF vectorization.
    
    Returns:
        desc_features: Array of feature vectors for each job by index
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Extract unique job titles and descriptions
    job_desc_dict = {}
    
    for _, row in kartierung_df.iterrows():
        job_title = row['new_job_title_en_occ']
        job_desc = row['new_job_description_en_occ']
        
        # Skip if already processed or description is missing
        if job_title in job_desc_dict or pd.isna(job_desc):
            continue
            
        job_desc_dict[job_title] = job_desc
    
    # Create TF-IDF vectors for job descriptions
    vectorizer = TfidfVectorizer(max_features=100)  # Limit features for efficiency
    
    # Get job titles in vocabulary order
    job_titles = [job for job, _ in sorted(job_vocab.items(), key=lambda x: x[1])]
    
    # Get descriptions for each job title (or empty string if missing)
    descriptions = [job_desc_dict.get(job, "") for job in job_titles]
    
    # Fit and transform descriptions
    desc_features = vectorizer.fit_transform(descriptions).toarray()
    
    # Add one additional row for padding index (0)
    padding_features = np.zeros((1, desc_features.shape[1]))
    
    # Add one additional row for unknown job index (len(job_vocab)+1)
    unknown_features = np.mean(desc_features, axis=0, keepdims=True)
    
    # Combine features
    final_features = np.vstack([padding_features, desc_features, unknown_features])
    
    return final_features

def sorted_career_sequence_for_person(person_id, df):
    person_df = df[df['_id'] == person_id]
    jobs = []
    for _, row in person_df.iterrows():
        sequence = row['experience_order']
        job_title = row['new_job_title_en_occ']
        jobs.append((sequence, job_title))
    
    # Sort by sequence and extract job titles
    sorted_jobs = [job for _, job in sorted(jobs, key=lambda x: x[0])]
    return sorted_jobs if len(sorted_jobs) >= 2 else []
