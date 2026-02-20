import os, random


if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set(''.join(docs)))

print(f"Unique chars: {uchars}")

BOS = len(uchars)
vocab_size = len(uchars) + 1

word = "emma"
tokens = [BOS] + [uchars.index(ch) for ch in word] + [BOS]

print(f"Tokens for {word}: {tokens}")
