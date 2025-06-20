import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import torch
import torch.nn as nn
import random
import string

MAX_LENGTH = 64
vocab = list(string.printable)
vocab_size = len(vocab)
char_to_idx = {ch: idx for idx, ch in enumerate(vocab)}
idx_to_char = {idx: ch for idx, ch in enumerate(vocab)}

def text_to_tensor(text, max_length=MAX_LENGTH):
    text = text[:max_length]
    text = text.ljust(max_length)
    indices = [char_to_idx.get(ch, 0) for ch in text]
    return torch.tensor(indices, dtype=torch.long)

def tensor_to_text(tensor):
    indices = tensor.cpu().numpy().tolist()
    return ''.join([idx_to_char.get(idx, '') for idx in indices])

def tamper_text_message(original_text, tamper_strength=0.1):
    """Randomly modifies a fraction of characters in the original text."""
    text_length = len(original_text)
    num_tampered_chars = int(text_length * tamper_strength)
    tampered_indices = random.sample(range(text_length), num_tampered_chars)
    tampered_text = list(original_text)
    for idx in tampered_indices:
        tampered_text[idx] = random.choice(string.printable)
    return ''.join(tampered_text)

# Model
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, seq_length):
        super(TextAutoencoder, self).__init__()
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_length * embed_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, seq_length * embed_dim),
            nn.LeakyReLU(0.2)
        )
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        embed = self.embedding(x)
        encoded = self.encoder(embed)
        decoded = self.decoder(encoded)
        decoded = decoded.view(-1, self.seq_length, self.embed_dim)
        logits = self.output_layer(decoded)
        return logits

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_dim = 32
hidden_dim = 64
seq_length = MAX_LENGTH

model = TextAutoencoder(vocab_size, embed_dim, hidden_dim, seq_length)
model.load_state_dict(torch.load("text_autoencoder_final.pth", map_location=device))
model.to(device)
model.eval()

chat_history = {
    "sender": [],
    "attacker": [],
    "receiver": []
}
senderData = {}       # to store sender's original message
tamperedData = {}     # to store tampered decrypted message

# Functions for Chat UI
def append_message(text, role, msg_type="system"):
    """Appends a message to the chat history and updates the display for the given role."""
    chat_history[role].append((msg_type, text))
    update_chat_display(role)

def update_chat_display(role):
    """Updates the chat display for the specified role."""
    if role == "sender":
        text_widget = sender_text
    elif role == "attacker":
        text_widget = attacker_text
    else:
        text_widget = receiver_text
    text_widget.config(state=tk.NORMAL)
    text_widget.delete("1.0", tk.END)
    for msg_type, msg in chat_history[role]:
        text_widget.insert(tk.END, f"{msg}\n")
    text_widget.config(state=tk.DISABLED)

# Functions for Model Operations
def send_message():
    message = sender_entry.get().strip()
    if not message:
        messagebox.showerror("Error", "Please enter a message.")
        return
    append_message("Sender: " + message, "sender", "sender")
    sender_entry.delete(0, tk.END)
    padded_message = message.ljust(MAX_LENGTH)
    tensor = text_to_tensor(padded_message).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(tensor)
    global senderData
    senderData = {"original": message}
    append_message("Message sent successfully.", "sender", "system")

def simulate_tampering():
    if not senderData.get("original"):
        messagebox.showerror("Error", "No sender message available to tamper with.")
        return
    orig_message = senderData["original"]
    tamper_strength = tamper_slider.get()  # slider value between 0 and 1
    padded_message = orig_message.ljust(MAX_LENGTH)
    clean_tensor = text_to_tensor(padded_message).unsqueeze(0).to(device)
    with torch.no_grad():
        clean_logits = model(clean_tensor)
        clean_pred = torch.argmax(clean_logits, dim=2).squeeze(0)
        original_text = tensor_to_text(clean_pred)
    # Creating tampered input using the tamper_text_message function
    tampered_input = tamper_text_message(orig_message, tamper_strength=tamper_strength)
    tampered_tensor = text_to_tensor(tampered_input).unsqueeze(0).to(device)
    with torch.no_grad():
        tampered_logits = model(tampered_tensor)
        tampered_pred = torch.argmax(tampered_logits, dim=2).squeeze(0)
        tampered_decrypted = tensor_to_text(tampered_pred)
    tampered_acc = (tampered_pred == text_to_tensor(orig_message).to(device)).float().mean().item()
    # For Attacker tab, displaying the original encrypted and the tampered input (as text)
    append_message("Original Encrypted: " + original_text, "attacker", "attacker")
    append_message("Tampered Encrypted: " + tampered_input, "attacker", "attacker")
    append_message("Message Tampered Successfully", "attacker", "system")
    # Saving tampered data for Receiver tab
    global tamperedData
    tamperedData = {"decrypted": tampered_decrypted}

def show_receiver():
    # When the Receiver tab is selected, displaying the decrypted message if available.
    if tamperedData.get("decrypted"):
        append_message("Decrypted Message: " + tamperedData["decrypted"], "receiver", "receiver")
        append_message("Message Decrypted Successfully", "receiver", "system")
    else:
        append_message("No tampered message received yet.", "receiver", "system")

# Tkinter UI
root = tk.Tk()
root.title("Neural Encryption Chat")
root.geometry("800x600")

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# Sender tab
sender_frame = ttk.Frame(notebook)
notebook.add(sender_frame, text="Sender")

sender_text = scrolledtext.ScrolledText(sender_frame, wrap=tk.WORD, state=tk.DISABLED, width=70, height=20)
sender_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

sender_entry = tk.Entry(sender_frame, font=("Helvetica", 14))
sender_entry.pack(padx=10, pady=(0,10), fill=tk.X)

send_button = tk.Button(sender_frame, text="Send", command=send_message, font=("Helvetica", 14), bg="#075e54", fg="white")
send_button.pack(padx=10, pady=(0,10))

# Attacker tab
attacker_frame = ttk.Frame(notebook)
notebook.add(attacker_frame, text="Attacker")

attacker_text = scrolledtext.ScrolledText(attacker_frame, wrap=tk.WORD, state=tk.DISABLED, width=70, height=20)
attacker_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

slider_frame = tk.Frame(attacker_frame)
slider_frame.pack(pady=5)

tamper_slider = tk.Scale(slider_frame, from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL, length=200)
tamper_slider.set(0.2)
tamper_slider.pack(side=tk.LEFT)

tamper_value_label = tk.Label(slider_frame, text="0.2", font=("Helvetica", 12))
tamper_value_label.pack(side=tk.LEFT, padx=5)

def slider_update(val):
    tamper_value_label.config(text=str(val))
tamper_slider.config(command=slider_update)

tamper_button = tk.Button(slider_frame, text="Simulate Tampering", command=simulate_tampering, font=("Helvetica", 12), bg="#00796b", fg="white")
tamper_button.pack(side=tk.LEFT, padx=10)

# Receiver tab
receiver_frame = ttk.Frame(notebook)
notebook.add(receiver_frame, text="Receiver")

receiver_text = scrolledtext.ScrolledText(receiver_frame, wrap=tk.WORD, state=tk.DISABLED, width=70, height=20)
receiver_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

def on_tab_change(event):
    selected_tab = notebook.index(notebook.select())
    if selected_tab == 2:
        show_receiver()

notebook.bind("<<NotebookTabChanged>>", on_tab_change)

root.mainloop()
