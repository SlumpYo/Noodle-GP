token_to_idx = {
    '<PAD>': 0,
    '<SOS>': 1,
    '<EOS>': 2,
}

# Build the list of all (string, fret, state) combinations
# Strings are 1–6, frets are 0–20, state is 'S' (sustain) or 'O' (onset)
triples = [
    (s, f, m)
    for s in range(1, 7)
    for f in range(0, 21)
    for m in ('S', 'O')
]

# Turn those into tokens and assign indices starting at 3
for idx, (s, f, m) in enumerate(triples, start=3):
    token = f's{s}_f{f}_{m}'
    token_to_idx[token] = idx

# Build the inverse map and vocab size
idx_to_token = {i: t for t, i in token_to_idx.items()}
vocab_size   = len(token_to_idx)