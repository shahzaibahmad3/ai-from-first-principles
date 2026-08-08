"""
Where does a transformer actually keep its parameters?

"Attention Is All You Need" is a title about what attention replaced -- recurrence
and convolution -- but it is widely read as though attention were the whole
computation. It isn't, and the accounting is simple enough to do exactly.

Per layer, with model width d:

    attention   Wq, Wk, Wv, Wo, each d x d              ->  4d²
    feed-forward  up d -> 4d, down 4d -> d              ->  8d²

So the feed-forward sublayer holds twice what attention holds, and that ratio does
not depend on d at all -- it survives every scale-up.

Modern models changed the feed-forward layer (gated SwiGLU, three matrices instead
of two) and changed attention (grouped-query, fewer key/value heads). Both moves
push in the SAME direction, so the headline is conservative. Printed below so the
claim is bounded by arithmetic rather than by assumption.

    python3 src/budget.py
"""


def attention_params(d, n_heads=None, n_kv_heads=None):
    """Wq, Wk, Wv, Wo. With grouped-query attention the K and V projections
    shrink by the ratio of key/value heads to query heads."""
    if n_heads is None or n_kv_heads is None or n_kv_heads == n_heads:
        return 4 * d * d
    head_dim = d // n_heads
    q = d * d
    k = d * (n_kv_heads * head_dim)
    v = d * (n_kv_heads * head_dim)
    o = d * d
    return q + k + v + o


def ffn_params(d, mult=4, gated=False):
    """Classic: two matrices, d->mult*d->d. Gated (SwiGLU): three matrices."""
    hidden = int(round(mult * d))
    return 3 * d * hidden if gated else 2 * d * hidden


MODELS = [
    # name,                d,     layers, ffn mult, gated, heads, kv heads
    ("GPT-3 175B",        12288,  96,     4,        False,  96,    96),
    ("GPT-2 1.5B",         1600,  48,     4,        False,  25,    25),
    ("Llama-2 7B",         4096,  32,     2.6875,   True,   32,    32),
    ("Llama-2 70B (GQA)",  8192,  80,     3.5,      True,   64,     8),
]


def report():
    rows = []
    for name, d, layers, mult, gated, heads, kv in MODELS:
        attn = attention_params(d, heads, kv)
        ffn = ffn_params(d, mult, gated)
        rows.append(dict(name=name, d=d, layers=layers, gated=gated,
                         attn_layer=attn, ffn_layer=ffn,
                         attn_total=attn * layers, ffn_total=ffn * layers,
                         ratio=ffn / attn, ffn_share=ffn / (attn + ffn)))
    return rows


if __name__ == "__main__":
    print("the exact per-layer arithmetic, classic architecture")
    print("  attention    Wq, Wk, Wv, Wo   4 x d x d  =  4d²")
    print("  feed-forward d->4d, 4d->d     2 x 4 x d² =  8d²")
    print("  ratio        8d² / 4d²        =  2 : 1, for any d\n")

    rows = report()
    print(f"  {'model':20s} {'attention':>11s} {'feed-forward':>13s} "
          f"{'FFN share':>10s} {'ratio':>8s}")
    print("  " + "-" * 68)
    for r in rows:
        tag = " (gated)" if r["gated"] else ""
        print(f"  {r['name']:20s} {r['attn_total']/1e9:>10.1f}B "
              f"{r['ffn_total']/1e9:>12.1f}B {r['ffn_share']*100:>9.1f}% "
              f"{r['ratio']:>7.2f}:1{tag}")

    g3 = rows[0]
    print(f"\nGPT-3, the model part 1 quoted: {g3['attn_total']/1e9:.1f}B in attention, "
          f"{g3['ffn_total']/1e9:.1f}B in the feed-forward layers,")
    print(f"together {(g3['attn_total']+g3['ffn_total'])/1e9:.1f}B of about 175B. "
          f"Attention is {(1-g3['ffn_share'])*100:.0f}% of it.")

    neurons = 4 * g3["d"] * g3["layers"]
    print(f"\nAnd those feed-forward layers are where the neurons are: "
          f"4d x layers = {neurons:,}")
    print("which is the 'about five million neurons' from part 1. Same units, all")
    print("the way back to the umbrella.")

    print("\nEvery modern change pushes the same way: gated feed-forward layers use")
    print("three matrices instead of two, and grouped-query attention shrinks the")
    print("key and value projections. So 'attention is a third of the parameters'")
    print("is the generous reading -- in the Llama rows above it is less.")
