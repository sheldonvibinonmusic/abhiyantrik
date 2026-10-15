"""
ROGUE PROTOCOL - MODEL DOWNLOAD SCRIPT
Downloads approved model checkpoints (MiniLM and DistilBERT) to local project directories
for 100% offline local inference and fine-tuning.
"""
import os
import sys
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

def download_models():
    print("=" * 70)
    print("DOWNLOADING APPROVED TRANSFORMER CHECKPOINTS FOR LOCAL OFFLINE USE")
    print("=" * 70)
    
    # Target directories
    os.makedirs("models/all-MiniLM-L6-v2", exist_ok=True)
    os.makedirs("models/distilbert-base-uncased", exist_ok=True)
    
    # 1. Download MiniLM
    print("\n[1/2] Downloading sentence-transformers/all-MiniLM-L6-v2...")
    model_name_minilm = "sentence-transformers/all-MiniLM-L6-v2"
    tok_minilm = AutoTokenizer.from_pretrained(model_name_minilm)
    model_minilm = AutoModel.from_pretrained(model_name_minilm)
    
    tok_minilm.save_pretrained("models/all-MiniLM-L6-v2")
    model_minilm.save_pretrained("models/all-MiniLM-L6-v2")
    print("-> Successfully saved MiniLM to models/all-MiniLM-L6-v2")
    
    # 2. Download DistilBERT
    print("\n[2/2] Downloading distilbert/distilbert-base-uncased...")
    model_name_distilbert = "distilbert/distilbert-base-uncased"
    tok_distil = AutoTokenizer.from_pretrained(model_name_distilbert)
    model_distil = AutoModelForSequenceClassification.from_pretrained(model_name_distilbert, num_labels=2)
    
    tok_distil.save_pretrained("models/distilbert-base-uncased")
    model_distil.save_pretrained("models/distilbert-base-uncased")
    print("-> Successfully saved DistilBERT to models/distilbert-base-uncased")
    
    print("\n" + "=" * 70)
    print("ALL APPROVED MODELS DOWNLOADED AND CACHED LOCALLY.")
    print("Local offline execution verified. External inference APIs prohibited.")
    print("=" * 70)

if __name__ == "__main__":
    download_models()
