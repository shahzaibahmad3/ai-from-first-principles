"""
A single neuron, written to be read.

This is the exact unit an LLM is built from -- in its simplest, classic form
(the "perceptron"): a weighted sum, then a yes/no. The two functions below are
23 lines of logic between them, no libraries. It's the same algorithm the
interactive playground runs. If you understand this file, you understand the
atom the whole field is made of.

Run it:
    python3 neuron.py
"""


def predict(weights, bias, point):
    """
    The forward pass: does the neuron fire for this point?

    Multiply each input by its weight, add them up, add the bias, and check
    the sign. Positive -> fire (1). Zero or negative -> stay quiet (0).
    That single number, `total`, is the neuron's entire opinion.
    """
    total = bias
    for w, x in zip(weights, point):
        total += w * x
    return 1 if total > 0 else 0


def train(points, labels, lr=0.1, epochs=100):
    """
    Rosenblatt's learning rule (1958).

    Start with a flat line (all weights zero). Walk through the examples one at
    a time. For each, guess. If the guess is right, change nothing. If it's
    wrong, nudge every weight by (learning_rate * error * input) and the bias
    by (learning_rate * error), where error is +1 or -1.

    That nudge is the whole idea: it moves the line a small step in the
    direction that would have gotten *this* point right. Do that enough times
    and -- if a separating line exists at all -- the line walks into place and
    the misses stop.
    """
    weights = [0.0] * len(points[0])
    bias = 0.0

    for epoch in range(epochs):
        misses = 0
        for point, target in zip(points, labels):
            guess = predict(weights, bias, point)
            error = target - guess          # +1, 0, or -1
            if error != 0:
                for i in range(len(weights)):
                    weights[i] += lr * error * point[i]
                bias += lr * error
                misses += 1

        # A full pass with zero misses means the line separates the data.
        # Nothing will change after this, so we can stop early.
        if misses == 0:
            print(f"separated after {epoch} passes")
            return weights, bias

    print(f"gave up after {epochs} passes -- still missing")
    return weights, bias


def report(name, points, labels, lr=0.1, epochs=100):
    """Train on one gate and print what the neuron ended up believing."""
    print(f"=== {name} ===")
    weights, bias = train(points, labels, lr=lr, epochs=epochs)

    print(f"\nlearned:  w1={weights[0]:+.2f}  w2={weights[1]:+.2f}  b={bias:+.2f}")
    print(f"line:     {weights[0]:+.2f}*x1 {weights[1]:+.2f}*x2 {bias:+.2f} = 0\n")

    print("  x1  x2 | want  got")
    print("  ---------|---------")
    wrong = 0
    for point, target in zip(points, labels):
        got = predict(weights, bias, point)
        if got != target:
            wrong += 1
        mark = "" if got == target else "  <- wrong"
        print(f"  {point[0]:+d}  {point[1]:+d} |   {target}    {got}{mark}")
    print(f"\n  {wrong} of {len(points)} wrong\n")


if __name__ == "__main__":
    # (-1 = off, +1 = on)
    points = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    # AND: fire only when BOTH inputs are on. Linearly separable, so the
    # perceptron is guaranteed to find a line -- and it does, quickly.
    report("AND gate (separable)", points, [0, 0, 0, 1])

    # XOR: fire when the inputs DIFFER. No straight line can do this, so
    # training burns all 100 passes and still gets one point wrong, forever.
    # That failure is the whole subject of the next post.
    report("XOR gate (impossible for one neuron)", points, [0, 1, 1, 0])
