"""
Text Generator Runner
Uses a character-level LSTM model to generate text.
"""

import torch
import torch.nn as nn
import os

# ============================================
# MODEL ARCHITECTURE (Character-Level LSTM)
# ============================================
class CharLSTM(nn.Module):
    """Character-level LSTM for text generation."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.0):
        super(CharLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)


# ============================================
# TEXT GENERATOR CLASS
# ============================================
class TextGenerator:
    """Handles loading the model and generating text."""
    
    def __init__(self, model_path, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.vocab_size = None
        
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load the trained model and vocabulary."""
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Check what's in the checkpoint
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"Checkpoint keys: {keys}")
            
            # Try to extract vocabulary
            if 'char_to_idx' in checkpoint:
                self.char_to_idx = checkpoint['char_to_idx']
                self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
            elif 'vocab' in checkpoint:
                self.char_to_idx = checkpoint['vocab']
                self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
            else:
                # Infer vocabulary size from embedding layer
                if 'embedding.weight' in checkpoint:
                    vocab_size = checkpoint['embedding.weight'].shape[0]
                    print(f"[!] No vocabulary found. Inferred vocab size: {vocab_size}")
                    # Create English-only ASCII vocabulary (letters, digits, punctuation)
                    # This is a guess - proper output requires the original vocabulary
                    ascii_chars = (
                        'abcdefghijklmnopqrstuvwxyz'
                        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                        '0123456789'
                        ' .,!?;:\'"()-\n\t'
                    )
                    # Repeat ASCII chars to fill vocab size
                    chars = list(ascii_chars)
                    while len(chars) < vocab_size:
                        chars.extend(list(ascii_chars))
                    chars = chars[:vocab_size]
                    self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
                    self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
                else:
                    # Fallback ASCII vocabulary
                    print("[!] No vocabulary found. Using default ASCII characters.")
                    chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?\n'
                    self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
                    self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
            
            self.vocab_size = checkpoint['embedding.weight'].shape[0] if 'embedding.weight' in checkpoint else len(self.char_to_idx)
            
            # Infer model parameters from checkpoint
            if 'embedding.weight' in checkpoint:
                embedding_dim = checkpoint['embedding.weight'].shape[1]
                hidden_dim = checkpoint['lstm.weight_hh_l0'].shape[1]
                
                # Count LSTM layers
                num_layers = 1
                while f'lstm.weight_ih_l{num_layers}' in checkpoint:
                    num_layers += 1
                
                print(f"Inferred model params: vocab={self.vocab_size}, embed={embedding_dim}, hidden={hidden_dim}, layers={num_layers}")
            else:
                # Default parameters
                embedding_dim = 128
                hidden_dim = 256
                num_layers = 1
            
            # Initialize model
            self.model = CharLSTM(
                vocab_size=self.vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers
            )
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # The checkpoint might be just the state dict
                self.model.load_state_dict(checkpoint)
        else:
            # Checkpoint is just the state dict
            print("[!] Checkpoint appears to be just a state dict.")
            print("[!] Using default vocabulary and model parameters.")
            
            chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
            self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
            self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
            self.vocab_size = len(self.char_to_idx)
            
            self.model = CharLSTM(vocab_size=self.vocab_size)
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"[OK] Model loaded successfully! Vocabulary size: {self.vocab_size}")
    
    def generate(self, seed_text, length=200, temperature=0.8):
        """
        Generate text starting from seed_text.
        
        Args:
            seed_text: Starting text for generation
            length: Number of characters to generate
            temperature: Controls randomness (higher = more random, lower = more deterministic)
        
        Returns:
            Generated text string
        """
        self.model.eval()
        
        # Convert seed text to indices
        chars = [ch for ch in seed_text if ch in self.char_to_idx]
        if not chars:
            print("[!] No valid characters in seed text. Using default seed.")
            chars = [' ']
        
        # Initialize hidden state
        hidden = self.model.init_hidden(1, self.device)
        
        # Process seed text
        for ch in chars[:-1]:
            idx = self.char_to_idx[ch]
            input_tensor = torch.tensor([[idx]]).to(self.device)
            _, hidden = self.model(input_tensor, hidden)
        
        # Start generating from last character
        current_char = chars[-1]
        generated = seed_text
        
        with torch.no_grad():
            for _ in range(length):
                idx = self.char_to_idx.get(current_char, 0)
                input_tensor = torch.tensor([[idx]]).to(self.device)
                output, hidden = self.model(input_tensor, hidden)
                
                # Apply temperature
                output = output.squeeze() / temperature
                probs = torch.softmax(output, dim=-1)
                
                # Sample from distribution
                next_idx = torch.multinomial(probs, 1).item()
                next_char = self.idx_to_char.get(next_idx, ' ')
                
                generated += next_char
                current_char = next_char
        
        return generated


# ============================================
# MAIN EXECUTION
# ============================================
def main():
    # Get the model path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "text generator.pth")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}")
        print("Please make sure 'text generator.pth' is in the same directory as this script.")
        return
    
    try:
        # Initialize generator
        generator = TextGenerator(model_path)
        
        print("\n" + "="*50)
        print("TEXT GENERATOR READY!")
        print("="*50)
        
        # Interactive loop
        while True:
            print("\nOptions:")
            print("  1. Generate text")
            print("  2. Exit")
            
            choice = input("\nEnter choice (1-2): ").strip()
            
            if choice == '1':
                seed = input("Enter seed text (or press Enter for default): ").strip()
                if not seed:
                    seed = "The "
                
                try:
                    length = int(input("Enter length to generate (default 200): ").strip() or "200")
                except ValueError:
                    length = 200
                
                try:
                    temp = float(input("Enter temperature 0.1-2.0 (default 0.8): ").strip() or "0.8")
                except ValueError:
                    temp = 0.8
                
                print("\nGenerating text...")
                generated = generator.generate(seed, length=length, temperature=temp)
                
                print("\n" + "-"*50)
                print("GENERATED TEXT:")
                print("-"*50)
                print(generated)
                print("-"*50)
            
            elif choice == '2':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        print("\n[!] This might happen if the model architecture doesn't match.")
        print("You may need to modify the CharLSTM class to match your trained model.")
        print(f"\nFull error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
