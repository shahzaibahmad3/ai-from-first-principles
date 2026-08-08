"""
A deep fully-connected network with every knob that mattered historically:
activation, initialisation scale, residual connections, normalisation.

Backpropagation is written out longhand -- no autograd -- because the whole
subject of this post is what happens to the gradient on its way back down, and
you cannot watch that if a library hides the backward pass.

The residual backward pass is the part worth reading carefully. A skip
connection means the layer's output is `f(x) + x`, so the gradient arriving at
that output takes *two* routes back: one through the weights, and one straight
through the identity. Miss the second and the whole thing quietly breaks --
which is exactly what happened in my first draft of this experiment.

    python3 src/depth.py        # gradient check + a smoke test
"""

import numpy as np

EPS = 1e-6


# ----------------------------------------------------------------- pieces
def sigmoid(z):
    """Stable on both tails: exp(-z) overflows for very negative z."""
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def rms_forward(x):
    """RMSNorm without the learned gain: x / sqrt(mean(x^2)).

    This is the normaliser modern LLMs actually use (Llama, Mistral, Gemma,
    Qwen), and unlike LayerNorm it does not re-centre -- which makes its
    backward pass short enough to read.
    """
    r = np.sqrt(np.mean(x * x, axis=1, keepdims=True) + EPS)
    return x / r, r


def rms_backward(dy, x, r):
    """d/dx of x/r. The second term is r's own dependence on every element."""
    d = x.shape[1]
    return dy / r - x * (np.sum(dy * x, axis=1, keepdims=True) / (d * r ** 3))


# Weight scales. "naive" is the N(0,1) that seemed reasonable before anyone had
# looked at what it does to signal variance; "he" accounts for ReLU zeroing half
# its inputs (He et al. 2015), which Xavier's linear-activation derivation does not.
SCALES = {
    "naive": lambda fan_in, fan_out: 1.0,
    "xavier": lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out)),
    "he": lambda fan_in, fan_out: np.sqrt(2.0 / fan_in),
}


class DeepNet:
    def __init__(self, depth, width, activation="sigmoid", init="naive",
                 residual=False, norm=False, n_in=2, seed=0):
        """`depth` counts weight matrices, so depth=2 is one hidden layer."""
        self.depth, self.width = depth, width
        self.activation, self.init = activation, init
        self.residual, self.norm = residual, norm

        rng = np.random.default_rng(seed)
        dims = [n_in] + [width] * (depth - 1) + [1]
        scale = SCALES[init]
        self.W = [rng.normal(0, scale(dims[i], dims[i + 1]), (dims[i], dims[i + 1]))
                  for i in range(depth)]
        self.b = [np.zeros(dims[i + 1]) for i in range(depth)]

    # ------------------------------------------------------------- forward
    def forward(self, X):
        """Returns the output plus every intermediate the backward pass needs."""
        a_in, z_of, raw_of, skipped, pre_norm, rms_r = [], [], [], [], [], []
        a = X
        for i in range(self.depth):
            a_in.append(a)
            z = a @ self.W[i] + self.b[i]
            z_of.append(z)

            if i == self.depth - 1:
                # The output layer is always a plain sigmoid: one probability.
                raw_of.append(None); skipped.append(False)
                pre_norm.append(None); rms_r.append(None)
                a = sigmoid(z)
                continue

            raw = sigmoid(z) if self.activation == "sigmoid" else np.maximum(z, 0.0)
            raw_of.append(raw)

            # A skip only makes sense where the shapes already match.
            skip = self.residual and raw.shape == a.shape
            skipped.append(skip)
            out = raw + a if skip else raw

            if self.norm and out.shape[1] == self.width:
                pre_norm.append(out)
                out, r = rms_forward(out)
                rms_r.append(r)
            else:
                pre_norm.append(None); rms_r.append(None)
            a = out

        return a, dict(a_in=a_in, z=z_of, raw=raw_of, skipped=skipped,
                       pre_norm=pre_norm, rms_r=rms_r)

    # ------------------------------------------------------------ backward
    def backward(self, X, y, cache=None, out=None):
        """Gradients of mean squared error. Returns (dW, db, per-layer |dW|)."""
        if cache is None:
            out, cache = self.forward(X)
        n = X.shape[0]
        dW = [None] * self.depth
        db = [None] * self.depth

        # dL/d(output) for L = mean((out - y)^2). The factor of 2 is real; it is
        # very often dropped and absorbed into the learning rate instead, but
        # keeping it means these gradients are exactly the gradients of the loss
        # this class reports -- which is what makes the gradient check meaningful.
        g = 2.0 * (out - y[:, None]) / n

        for i in range(self.depth - 1, -1, -1):
            if i == self.depth - 1:
                dz = g * out * (1.0 - out)                 # output sigmoid
            else:
                if cache["pre_norm"][i] is not None:       # undo the normaliser
                    g = rms_backward(g, cache["pre_norm"][i], cache["rms_r"][i])
                if self.activation == "sigmoid":
                    raw = cache["raw"][i]
                    dz = g * raw * (1.0 - raw)
                else:
                    dz = g * (cache["z"][i] > 0)

            dW[i] = cache["a_in"][i].T @ dz
            db[i] = dz.sum(axis=0)

            g_next = dz @ self.W[i].T
            if cache["skipped"][i]:
                g_next = g_next + g                        # the identity route
            g = g_next

        return dW, db, [float(np.linalg.norm(m)) for m in dW]

    # ------------------------------------------------------------- helpers
    def loss(self, X, y):
        out, _ = self.forward(X)
        return float(np.mean((out - y[:, None]) ** 2))

    def accuracy(self, X, y):
        out, _ = self.forward(X)
        return float(((out[:, 0] > 0.5).astype(float) == y).mean())

    def grad_norms(self, X, y):
        return self.backward(X, y)[2]

    def fit(self, X, y, epochs, lr, momentum=0.9):
        """Plain momentum gradient descent -- deliberately not an adaptive
        optimiser, so that what we see is the raw gradient's doing."""
        vW = [np.zeros_like(w) for w in self.W]
        vb = [np.zeros_like(bb) for bb in self.b]
        for _ in range(epochs):
            out, cache = self.forward(X)
            dW, db, _ = self.backward(X, y, cache, out)
            for i in range(self.depth):
                vW[i] = momentum * vW[i] + dW[i]
                vb[i] = momentum * vb[i] + db[i]
                self.W[i] -= lr * vW[i]
                self.b[i] -= lr * vb[i]
        return self


# ------------------------------------------------------------------ data
def rings(n=240, seed=0):
    """An inner blob inside an outer annulus. No straight line separates it, and
    unlike XOR two hidden units genuinely cannot bend far enough for it."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(0, 2 * np.pi, n)
    inner = np.arange(n) % 2 == 0
    rad = np.where(inner, rng.uniform(0.05, 0.28, n), rng.uniform(0.45, 0.70, n))
    X = np.stack([0.5 + rad * np.cos(ang), 0.5 + rad * np.sin(ang)], axis=1)
    return X, inner.astype(float)


CONFIGS = {
    "sigmoid":          dict(activation="sigmoid", init="naive", residual=False, norm=False),
    "relu_he":          dict(activation="relu",    init="he",    residual=False, norm=False),
    "relu_he_res":      dict(activation="relu",    init="he",    residual=True,  norm=False),
    "relu_he_res_norm": dict(activation="relu",    init="he",    residual=True,  norm=True),
}

LABELS = {
    "sigmoid":          "sigmoid, N(0,1) init",
    "relu_he":          "ReLU + He init",
    "relu_he_res":      "ReLU + He + residual",
    "relu_he_res_norm": "ReLU + He + residual + RMSNorm",
}


# ------------------------------------------------------------ self-check
def _gradient_check():
    """Finite differences against the analytic gradient. This is here because a
    hand-written backward pass through skips and a normaliser is very easy to
    get subtly wrong, and a subtly wrong gradient still trains -- just worse."""
    X, y = rings(n=24, seed=3)
    worst = 0.0
    for name, cfg in CONFIGS.items():
        for depth in (2, 5):
            net = DeepNet(depth=depth, width=6, seed=1, **cfg)
            dW, _, _ = net.backward(X, y)
            h = 1e-6
            for i in range(depth):
                idx = (0, 0)
                before = net.W[i][idx]
                net.W[i][idx] = before + h; lp = net.loss(X, y)
                net.W[i][idx] = before - h; lm = net.loss(X, y)
                net.W[i][idx] = before
                numeric = (lp - lm) / (2 * h)
                analytic = dW[i][idx]
                denom = max(abs(numeric), abs(analytic), 1e-12)
                rel = abs(numeric - analytic) / denom
                worst = max(worst, rel)
                assert rel < 2e-4, (
                    f"{name} depth={depth} layer {i}: analytic {analytic:.3e} "
                    f"vs numeric {numeric:.3e} (rel {rel:.2e})")
    print(f"gradient check passed for all {len(CONFIGS)} configs "
          f"(worst relative error {worst:.2e})")


if __name__ == "__main__":
    _gradient_check()

    X, y = rings()
    print("\nsmoke test: depth 20, 4000 epochs, lr 0.05")
    for name, cfg in CONFIGS.items():
        net = DeepNet(depth=20, width=16, seed=1, **cfg)
        att0 = net.grad_norms(X, y)
        net.fit(X, y, epochs=4000, lr=0.05)
        print(f"  {LABELS[name]:34s} acc {net.accuracy(X, y)*100:5.1f}%   "
              f"layer-1 |grad| at init {att0[0]:.3e}")
