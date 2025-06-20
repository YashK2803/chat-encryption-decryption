# Chat Encryption Decryption

## Neural Encryption Decryption
A secure message transmission simulator built with Tkinter GUI and a PyTorch-based autoencoder model. Users can act as a Sender, Attacker, or Receiver, tamper with encrypted messages, and visually observe decrypted results.

## Overview
Neural Encryption Chat demonstrates how neural networks can be applied to simulate encryption and decryption in communication. The tool includes:
• **Sender**: Inputs a plaintext message that gets encoded using a trained autoencoder.  
• **Attacker**: Modifies the message with a tampering slider and sends it forward.  
• **Receiver**: Attempts to decode the original message, with feedback on tampering.

## Requirements

### Python Version
• Python 3.7+

### Dependencies
- torch >=1.10.0
- numpy
- nltk
- tk

Make sure to download NLTK's English word corpus:
```python
import nltk
nltk.download('words')
```

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python main.py
```

## Model Details

### Architecture
• Character-level autoencoder with embedding and MLP layers.  
• Input is tampered at configurable strength before decoding.  
• Accuracy metrics show resistance to tampering.

### Training
• Trained on synthetic English word dataset (from NLTK).  
• Messages are padded to a fixed length of 64 characters.  
• Trained with minor noise for robustness.

## Project Structure
```
encryption-decryption/
├── main.py                    # Tkinter GUI application
├── text_autoencoder_final.pth # Pretrained model weights
├── Training.ipynb             # Model Training Notebook
├── requirements.txt
└── README.pdf
```

## Features
• Graphical interface for interacting as sender, attacker, or receiver  
• Real-time tampering and decryption feedback  
• Synthetic message training for generalizability  
• Accuracy display after tampered decryption