"""
Does attention know what order the tokens are in?

Nothing here trains. The test is a single question asked three ways: shuffle the
input tokens, and see whether the output is the same set of vectors in the same
shuffled order. If it is, the mechanism cannot possibly be using position -- it
is computing on a bag of tokens, not a sequence.

Run three configurations, because the answer is different for each and the
difference is the interesting part:

  plain               just Q, K, V -- the mechanism on its own
  + positional        the sinusoids from Vaswani et al. 2017 added to the input
  + causal mask       each token may only look at itself and earlier ones

    python3 src/permutation.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from attention import D_MODEL, attend, positional_encoding

T = 6           # tokens
D_K = 8
N_TRIALS = 200  # independent random inputs, weights and permutations


def max_output_shift(rng, use_pos, causal):
    """How far the output moves when the input tokens are shuffled.

    Zero means: shuffling the input merely shuffles the output, so the mechanism
    is blind to order. Anything else means order changed the answer.
    """
    X = rng.normal(0, 1, (T, D_MODEL))
    Wq, Wk, Wv = (rng.normal(0, 0.5, (D_MODEL, D_K)) for _ in range(3))
    perm = rng.permutation(T)

    Xin = X + positional_encoding(T, D_MODEL) if use_pos else X
    # Shuffling the *tokens* means the positional signal stays where it is --
    # position 0 is still position 0. That is the whole point of adding it.
    Xin_p = X[perm] + positional_encoding(T, D_MODEL) if use_pos else X[perm]

    out, A = attend(Xin, Wq, Wk, Wv, causal=causal)
    out_p, A_p = attend(Xin_p, Wq, Wk, Wv, causal=causal)

    out_shift = float(np.abs(out_p - out[perm]).max())
    attn_shift = float(np.abs(A_p - A[np.ix_(perm, perm)]).max())
    return out_shift, attn_shift


CASES = [
    ("plain attention",       dict(use_pos=False, causal=False)),
    ("+ positional encoding", dict(use_pos=True,  causal=False)),
    ("+ causal mask",         dict(use_pos=False, causal=True)),
]


def run(n_trials=N_TRIALS, seed=0):
    rng = np.random.default_rng(seed)
    results = {}
    for label, kw in CASES:
        outs, attns = [], []
        for _ in range(n_trials):
            o, a = max_output_shift(rng, **kw)
            outs.append(o); attns.append(a)
        results[label] = dict(out_max=max(outs), out_median=float(np.median(outs)),
                              attn_max=max(attns))
    return results


if __name__ == "__main__":
    res = run()
    print(f"shuffling the input tokens, worst case over {N_TRIALS} random trials\n")
    print(f"  {'configuration':24s} {'max output shift':>18s} {'max attn shift':>16s}")
    print("  " + "-" * 60)
    for label, _ in CASES:
        r = res[label]
        print(f"  {label:24s} {r['out_max']:>18.3e} {r['attn_max']:>16.3e}")

    eps = np.finfo(np.float64).eps
    print(f"\n  float64 machine epsilon is {eps:.3e}, for scale.")
    print("\nPlain attention is not approximately order-blind. It is order-blind to")
    print("the last bit of a double: shuffling the tokens shuffles the output rows")
    print("and changes nothing else, and the attention matrix is the very same")
    print("matrix with its rows and columns permuted.")
    print("\nThere is no index in the formula. A = softmax(Q Kᵀ / √d) compares every")
    print("token with every other token and never asks where either of them sits.")
    print("\nBoth of the other two rows are large, and that is the real lesson: a")
    print("transformer gets its sense of order from things bolted on beside the")
    print("mechanism. Positional encodings are the famous one. The causal mask is")
    print("the other, and it is easy to forget -- if a token may only look")
    print("backwards, then which tokens it can see IS positional information.")
