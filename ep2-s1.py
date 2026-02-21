"""
The most atomic way to train and inference a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

Now enhanced with:
- DEBUG printing
- EPISODE controlled early exits

@karpathy
"""

"""
====================================================================
MicroGPT – A Minimal Transformer Built in Pure Python
====================================================================

This file implements a complete GPT-style language model
using only core Python. No PyTorch. No NumPy. No GPUs.

It includes:
- Character-level tokenization
- Automatic differentiation (autograd)
- Multi-head self-attention
- Feed-forward network (MLP)
- RMS Normalization
- Residual connections
- Adam optimizer
- Model saving/loading
- Text generation with temperature

--------------------------------------------------------------------
TRANSFORMER BLOCK STRUCTURE
--------------------------------------------------------------------

            Input Token
                 │
                 ▼
        Token Embedding (wte)
                 │
                 ▼
        Positional Embedding (wpe)
                 │
                 ▼
             Add + RMSNorm
                 │
                 ▼
        ┌────────────────────┐
        │   Multi-Head       │
        │     Attention      │
        └────────────────────┘
                 │
           + Residual
                 │
                 ▼
        ┌────────────────────┐
        │   Feed Forward     │
        │      (MLP)         │
        └────────────────────┘
                 │
           + Residual
                 │
                 ▼
            Linear (lm_head)
                 │
                 ▼
              Logits
                 │
                 ▼
             Softmax
                 │
                 ▼
          Next Token Probabilities

--------------------------------------------------------------------
SELF-ATTENTION (Single Head) CONCEPT
--------------------------------------------------------------------

For each token:

   Q = xWq
   K = xWk
   V = xWv

Attention Scores:
   score = Q · Kᵀ / √d

Attention Weights:
   softmax(scores)

Output:
   weighted_sum(weights * V)

Multi-head = Do this multiple times in parallel.

--------------------------------------------------------------------
This is the complete algorithm.
Everything else in modern libraries is optimization.
====================================================================
"""

import os
import math
import random
random.seed(42)

import argparse
import pickle
import sys

# ============================================================
# GLOBAL DEBUG / EPISODE CONTROL
# ============================================================

EPISODE = 0      # 0 = full model
DEBUG = True     # set False to disable prints

# ============================================================
# SAVE / LOAD
# ============================================================

def save_model(state_dict, filepath="microgpt_weights.pkl"):
    raw_state = {}
    for k, mat in state_dict.items():
        raw_state[k] = [[v.data for v in row] for row in mat]
    with open(filepath, "wb") as f:
        pickle.dump(raw_state, f)
    print(f"Model saved to {filepath}")

def load_model(state_dict, filepath="microgpt_weights.pkl"):
    with open(filepath, "rb") as f:
        raw_state = pickle.load(f)

    for k in state_dict.keys():
        for i in range(len(state_dict[k])):
            for j in range(len(state_dict[k][i])):
                state_dict[k][i][j].data = raw_state[k][i][j]

    print(f"Model loaded from {filepath}")

def cosine_similarity(vec1, vec2):
    dot = sum(a.data * b.data for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a.data**2 for a in vec1))
    norm2 = math.sqrt(sum(b.data**2 for b in vec2))
    return dot / (norm1 * norm2 + 1e-8)


def show_similar_letters(letter, top_k=5):
    if letter not in uchars:
        print(f"Letter '{letter}' not in vocabulary.")
        return

    idx = uchars.index(letter)
    base_vec = state_dict['wte'][idx]

    similarities = []

    for i, ch in enumerate(uchars):
        vec = state_dict['wte'][i]
        sim = cosine_similarity(base_vec, vec)
        similarities.append((ch, sim))

    # Sort descending by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🔎 Letters most similar to '{letter}':\n")

    for ch, sim in similarities[:top_k]:
        print(f"{ch}  →  {sim:.4f}")

# ============================================================
# Dataset Loading
# ============================================================

"""
We train on a list of names.
Each line in input.txt is a separate training example.
"""

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)

print(f"num docs: {len(docs)}")

# ============================================================
# Tokenizer
# ============================================================

"""
Character-level tokenizer.

Each unique character becomes a token ID.
We also add a special BOS (Beginning Of Sequence) token.
"""

uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

print(f"vocab size: {vocab_size}")

# ============================================================
# AUTOGRAD ENGINE
# ============================================================

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1

    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)

        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ============================================================
# Model Configuration
# ============================================================

"""
n_embd     = embedding size
n_head     = number of attention heads
n_layer    = number of transformer blocks
block_size = maximum sequence length
"""

n_embd = 16
n_head = 4
n_layer = 1
block_size = 16
head_dim = n_embd // n_head

# ============================================================
# Parameter Initialization
# ============================================================

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    'wte': matrix(vocab_size, n_embd),
    'wpe': matrix(block_size, n_embd),
    'lm_head': matrix(vocab_size, n_embd)
}

for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4*n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4*n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

# ============================================================
# Helper Functions
# ============================================================

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)


    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    if DEBUG:
        print("\n--- RMSNORM DEBUG ---")
        print("Mean square:", round(ms.data, 6))
        print("Scale factor:", round(scale.data, 6))

    return [xi * scale for xi in x]

# ============================================================
# GPT FORWARD (WITH DEBUG + EPISODES)
# ============================================================
# ============================================================
# GPT Forward Pass
# ============================================================


def gpt(token_id, pos_id, keys, values):

    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]

    x = [t + p for t, p in zip(tok_emb, pos_emb)]

    if DEBUG and pos_id == 0:
        print("\n--- EMBEDDING DEBUG ---")
        print("Token ID:", token_id)
        print("Token Embedding (first 5 values):",
              [round(v.data, 4) for v in tok_emb[:5]])
        print("Position Embedding (first 5 values):",
              [round(v.data, 4) for v in pos_emb[:5]])
        print("Combined (first 5 values):",
              [round(v.data, 4) for v in x[:5]])

    # if DEBUG and pos_id == 0:
    #     print("\n[Episode 1] Embedding Output:", [round(v.data,4) for v in x[:5]])
    #     if EPISODE == 1: sys.exit()

    x = rmsnorm(x)

    if DEBUG and pos_id == 0:
        print("[Episode 2] After RMSNorm:", [round(v.data,4) for v in x[:5]])
        if EPISODE == 2: sys.exit()

    for li in range(n_layer):

        # ---------------------------
        # Multi-Head Attention
        # ---------------------------

        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])

        # if DEBUG and pos_id == 0:
        #     print("[Episode 3] Q sample:", [round(val.data,4) for val in q[:5]])
        if DEBUG and pos_id == 0:
            print("\n--- QKV DEBUG ---")
            print("Q (first 5):", [round(val.data,4) for val in q[:5]])
            print("K (first 5):", [round(val.data,4) for val in k[:5]])
            print("V (first 5):", [round(val.data,4) for val in v[:5]])
            if EPISODE == 3: sys.exit()

        keys[li].append(k)
        values[li].append(v)

        x_attn = []

        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]

            attn_logits = [
                sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                for t in range(len(k_h))
            ]

            attn_weights = softmax(attn_logits)

            # if DEBUG and pos_id == 0:
            # if DEBUG and pos_id == len(keys[li]) - 1:
            if DEBUG:
                print("\n--- ATTENTION DEBUG ---")
                print("Attention logits:",
                      [round(val.data,4) for val in attn_logits])
                print("Attention weights:",
                      [round(val.data,4) for val in attn_weights])
                # print("[Episode 4] Attention Weights:", [round(w.data,4) for w in attn_weights])
                if EPISODE == 4: sys.exit()

            head_out = [
                sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                for j in range(head_dim)
            ]

            x_attn.extend(head_out)

        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]

        if DEBUG:
            print(f"\nHead {h} output (first 3 vals):",
                  [round(val.data,4) for val in head_out[:3]])

        # if DEBUG and pos_id == 0:
            # print("[Episode 5] After Attention Residual:", [round(v.data,4) for v in x[:5]])
            if EPISODE == 5: sys.exit()

        # ---------------------------
        # MLP
        # ---------------------------

        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

        if DEBUG:
            print("\n--- RESIDUAL DEBUG ---")
            print("Before residual (first 3):",
                  [round(a.data,4) for a in x_residual[:3]])
            print("After residual (first 3):",
                  [round(a.data,4) for a in x[:3]])

        # if DEBUG and pos_id == 0:
        #     print("[Episode 6] After MLP:", [round(v.data,4) for v in x[:5]])
            if EPISODE == 6: sys.exit()

    logits = linear(x, state_dict['lm_head'])

    # if DEBUG:
    #     print("\n--- LOGITS DEBUG ---")
    #     print("Raw logits (first 5):",
    #           [round(val.data,4) for val in logits[:5]])
    #     if EPISODE == 8: sys.exit()

    # if DEBUG and pos_id == 0:
    #     print("[Episode 7] Logits sample:", [round(v.data,4) for v in logits[:5]])
    #     if EPISODE == 7: sys.exit()

    return logits

# ============================================================
# Training
# ============================================================

# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer
v = [0.0] * len(params) # second moment buffer


def train_model():
    num_steps = 1000

    for step in range(num_steps):

        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []

        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            if DEBUG and pos_id == 0:
                print("\n--- SOFTMAX DEBUG (Training) ---")
                print("Probabilities (first 5):",
                      [round(p.data, 4) for p in probs[:5]])
                if EPISODE == 8:
                    sys.exit()
            loss_t = -probs[target_id].log()
            losses.append(loss_t)

        loss = (1 / n) * sum(losses)

        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)

        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        if (step % 10 == 0):
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}")

    save_model(state_dict)


# ============================================================
# Inference
# ============================================================
def is_word_present(word, filename="input.txt"):
    with open(filename, "r", encoding="utf-8") as f:
        words = set(line.strip().lower() for line in f if line.strip())
    return "yes" if word.lower().strip() in words else "new"


def run_inference(temperature, n_samples=5):
    # print(f"\n--- inference (temperature={temperature}) ---")

    # for sample_idx in range(n_samples):
    #     keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    #     token_id = BOS
    #     sample = []

    #     for pos_id in range(block_size):
    #         logits = gpt(token_id, pos_id, keys, values)
    #         probs = softmax([l / temperature for l in logits])
    #         token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
    #         if token_id == BOS:
    #             break
    #         sample.append(uchars[token_id])

    #     print("sample:", ''.join(sample))

    print(f"\n--- inference (temperature = {temperature}) ---")

    yes_count = 0
    new_count = 0

    for sample_idx in range(n_samples):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = BOS
        sample = []

        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([l / temperature for l in logits])

            token_id = random.choices(
                range(vocab_size),
                weights=[p.data for p in probs]
            )[0]

            if token_id == BOS:
                break

            sample.append(uchars[token_id])

        word = ''.join(sample)
        ans = is_word_present(word)

        if ans == "yes":
            yes_count += 1
        else:
            new_count += 1

        print(f"sample {sample_idx+1:2d}: {word} - {ans}")

    # 🔥 Compute statistics
    total = yes_count + new_count
    novelty_percent = (new_count / total) * 100
    memorization_percent = (yes_count / total) * 100

    print("\n📊 Summary")
    print(f"Known words (memorized): {yes_count} ({memorization_percent:.2f}%)")
    print(f"New words (novel):       {new_count} ({novelty_percent:.2f}%)")

    return novelty_percent


# ============================================================
# MAIN - CLI
# ============================================================

# def main():
#     global EPISODE

#     parser = argparse.ArgumentParser()
#     parser.add_argument("-t", "--train", action="store_true")
#     parser.add_argument("-i", "--infer", action="store_true")
#     parser.add_argument("--temp", type=float, default=0.5)
#     parser.add_argument("--episode", type=int, default=0)

#     args = parser.parse_args()
#     EPISODE = args.episode

#     if args.infer:
#         load_model(state_dict)
#         run_inference(args.temp)
#     elif args.train:
#         print("Training not modified here. Use your existing training loop.")
#     else:
#         print("Specify -t or -i")

def main():
    global EPISODE

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", action="store_true")
    parser.add_argument("-i", "--infer", action="store_true")
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--sim", type=str, help="Show similar letters")

    args = parser.parse_args()
    EPISODE = args.episode

    if args.sim:
	    print("📦 Loading model...")
	    load_model(state_dict)
	    show_similar_letters(args.sim)
	    return

    if args.train:
        print("🚀 Training...")
        train_model()

    if args.infer:
        print("📦 Loading model...")
        load_model(state_dict)
        run_inference(args.temp)

    if not args.train and not args.infer:
        print("Specify -t and/or -i")


if __name__ == "__main__":
    main()
