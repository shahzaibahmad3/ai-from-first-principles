"""
One attention head, written to be read -- and to be checked.

Attention is usually introduced as a formula to memorise. It is a soft dictionary
lookup: every token proposes a query, every token advertises a key, the two are
compared, and the result is a weighted average of the values. That is all.

    A = softmax(Q Kᵀ / √d)      who to listen to
    out = A V                   listen to them

Backpropagation through it is written out longhand here -- no autograd -- because
the whole series is built on being able to see the arithmetic. Running this file
gradient-checks that backward pass against finite differences before it does
anything else, which is the habit part 3 taught: a hand-written gradient through a
softmax is easy to get subtly wrong, and a subtly wrong gradient still trains.

    python3 src/attention.py        # gradient check, then train on retrieval
"""

import numpy as np

# ------------------------------------------------------------------ task
# A tiny associative-recall problem, which is the cleanest thing a *single* head
# can be asked to do. Each item token carries a key and a value in one embedding;
# a final query token carries only a key. The head has to find the item whose key
# matches and report its value. If attention really is a lookup, one head is
# enough -- and we can check whether the weight lands on the right item.
N_KEYS = 6
N_VALUES = 6
N_ITEMS = 4
D_MODEL = N_KEYS + N_VALUES          # [ one-hot key | one-hot value ]


def make_batch(n, rng):
    """Returns (X, y, key_pos). key_pos is which item the query should match."""
    X = np.zeros((n, N_ITEMS + 1, D_MODEL))
    y = np.zeros(n, dtype=int)
    key_pos = np.zeros(n, dtype=int)
    for b in range(n):
        keys = rng.permutation(N_KEYS)[:N_ITEMS]      # distinct keys per sequence
        vals = rng.integers(0, N_VALUES, N_ITEMS)
        for t, (k, v) in enumerate(zip(keys, vals)):
            X[b, t, k] = 1.0
            X[b, t, N_KEYS + v] = 1.0
        j = rng.integers(0, N_ITEMS)
        X[b, N_ITEMS, keys[j]] = 1.0                  # the query: key only
        y[b] = vals[j]
        key_pos[b] = j
    return X, y, key_pos


# ------------------------------------------------------------- primitives
def softmax(z, axis=-1):
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def positional_encoding(t, d):
    """The original sinusoids (Vaswani et al. 2017). Nothing about attention
    itself knows about position; this is the signal that puts it there."""
    pos = np.arange(t)[:, None]
    i = np.arange(d)[None, :]
    ang = pos / np.power(10000.0, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(ang), np.cos(ang))


def attend(X, Wq, Wk, Wv, causal=False):
    """Forward pass only, for the permutation experiments. X is (T, d)."""
    t = X.shape[0]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(Wq.shape[1])
    if causal:
        scores = np.where(np.tril(np.ones((t, t), dtype=bool)), scores, -np.inf)
    A = softmax(scores, axis=-1)
    return A @ V, A


# ------------------------------------------------------------------ head
class Head:
    """One attention head plus a linear readout, trained with cross-entropy."""

    def __init__(self, d_k=8, seed=0, use_pos=False, causal=False):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(D_MODEL)
        self.Wq = rng.normal(0, s, (D_MODEL, d_k))
        self.Wk = rng.normal(0, s, (D_MODEL, d_k))
        self.Wv = rng.normal(0, s, (D_MODEL, d_k))
        self.Wo = rng.normal(0, 1.0 / np.sqrt(d_k), (d_k, N_VALUES))
        self.d_k = d_k
        self.use_pos = use_pos
        self.causal = causal

    def names(self):
        return ["Wq", "Wk", "Wv", "Wo"]

    def _inputs(self, X):
        if not self.use_pos:
            return X
        return X + positional_encoding(X.shape[1], D_MODEL)[None, :, :]

    def forward(self, X):
        Xin = self._inputs(X)
        Q, K, V = Xin @ self.Wq, Xin @ self.Wk, Xin @ self.Wv
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
        if self.causal:
            t = X.shape[1]
            scores = np.where(np.tril(np.ones((t, t), dtype=bool)), scores, -np.inf)
        A = softmax(scores, axis=-1)
        C = A @ V
        read = C[:, -1, :]                       # the query sits at the last position
        logits = read @ self.Wo
        return logits, dict(Xin=Xin, Q=Q, K=K, V=V, A=A, C=C, read=read)

    def loss_and_grads(self, X, y):
        n = X.shape[0]
        logits, c = self.forward(X)
        P = softmax(logits, axis=-1)
        loss = float(-np.log(np.maximum(P[np.arange(n), y], 1e-12)).mean())

        # cross-entropy through softmax collapses to (P - onehot)
        d_logits = P.copy()
        d_logits[np.arange(n), y] -= 1.0
        d_logits /= n

        g_Wo = c["read"].T @ d_logits
        d_read = d_logits @ self.Wo.T
        d_C = np.zeros_like(c["C"])
        d_C[:, -1, :] = d_read                   # only the query row was read

        A, V = c["A"], c["V"]
        g_V = A.transpose(0, 2, 1) @ d_C
        d_A = d_C @ V.transpose(0, 2, 1)
        # softmax jacobian, applied row-wise
        d_scores = A * (d_A - (d_A * A).sum(axis=-1, keepdims=True))
        d_scores /= np.sqrt(self.d_k)

        d_Q = d_scores @ c["K"]
        d_K = d_scores.transpose(0, 2, 1) @ c["Q"]
        Xt = c["Xin"].transpose(0, 2, 1)
        return loss, {
            "Wq": (Xt @ d_Q).sum(axis=0),
            "Wk": (Xt @ d_K).sum(axis=0),
            "Wv": (Xt @ g_V).sum(axis=0),
            "Wo": g_Wo,
        }

    def fit(self, steps, lr=0.5, batch=64, seed=0):
        rng = np.random.default_rng(seed)
        losses = []
        for _ in range(steps):
            X, y, _ = make_batch(batch, rng)
            loss, g = self.loss_and_grads(X, y)
            for name in self.names():
                setattr(self, name, getattr(self, name) - lr * g[name])
            losses.append(loss)
        return losses

    def evaluate(self, n=500, seed=999):
        """Accuracy, and how often the weight lands on the matching item."""
        X, y, key_pos = make_batch(n, np.random.default_rng(seed))
        logits, c = self.forward(X)
        acc = float((logits.argmax(-1) == y).mean())
        query_row = c["A"][:, -1, :N_ITEMS]
        points_at = float((query_row.argmax(-1) == key_pos).mean())
        return acc, points_at, c["A"]


# ------------------------------------------------------------ self-check
def _gradient_check():
    rng = np.random.default_rng(0)
    worst, where = 0.0, ""
    for use_pos in (False, True):
        for causal in (False, True):
            h = Head(seed=3, use_pos=use_pos, causal=causal)
            X, y, _ = make_batch(32, rng)
            _, g = h.loss_and_grads(X, y)
            for name in h.names():
                W = getattr(h, name)
                for _ in range(5):
                    i = rng.integers(0, W.shape[0])
                    j = rng.integers(0, W.shape[1])
                    eps, before = 1e-6, W[i, j]
                    W[i, j] = before + eps; lp = h.loss_and_grads(X, y)[0]
                    W[i, j] = before - eps; lm = h.loss_and_grads(X, y)[0]
                    W[i, j] = before
                    num = (lp - lm) / (2 * eps)
                    ana = g[name][i, j]
                    rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
                    if rel > worst:
                        worst, where = rel, f"{name} pos={use_pos} causal={causal}"
                    assert rel < 1e-3, (
                        f"gradient mismatch in {name} (pos={use_pos}, causal={causal}): "
                        f"analytic {ana:.3e} vs numeric {num:.3e}, rel {rel:.2e}")
    print(f"gradient check passed for all 4 variants "
          f"(worst relative error {worst:.2e}, at {where})")


if __name__ == "__main__":
    _gradient_check()

    print("\ntraining one head on associative recall "
          f"({N_ITEMS} items, {N_KEYS} keys, {N_VALUES} values)")
    print(f"  {'seed':>5}  {'accuracy':>9}  {'points at the matching key':>27}  {'final loss':>11}")
    print("  " + "-" * 60)
    for seed in (0, 1, 2):
        h = Head(seed=seed)
        losses = h.fit(steps=4000, lr=0.5, seed=100 + seed)
        acc, points_at, _ = h.evaluate()
        print(f"  {seed:>5}  {acc*100:>8.1f}%  {points_at*100:>26.1f}%  {losses[-1]:>11.4f}")

    print("\nOne head. No positional encoding, no mask, no feed-forward layer.")
    print("It finds the matching key and reads off its value, which is what the")
    print("formula says it should do -- and you can watch it happen, because the")
    print("attention weight is a number you can look at.")
