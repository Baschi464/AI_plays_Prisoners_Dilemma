def encode_state(history):     # history is a list of 5 tuples: (self_action, opp_action)
    """Encode last 5 (self, opp) moves as integer index ∈ [0, 1023]"""
    idx = 0
    for (a, b) in history:
        code = a * 2 + b
        idx = (idx << 2) | code
    return idx

def decode_state(idx):
    """Decode an integer index ∈ [0, 1023] into a 5-round history of (a, b) pairs"""
    history = []
    for _ in range(5):
        code = idx & 0b11  # last 2 bits
        b = code & 1
        a = (code >> 1) & 1
        history.insert(0, (a, b))
        idx >>= 2
    return history