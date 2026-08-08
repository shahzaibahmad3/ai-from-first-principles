# Stacking Was the Easy Part. Getting the Gradient Back Down Wasn't.

### Part 3 of *AI, from first principles*: why backpropagation existed for decades before depth worked, measured one layer at a time.

![Gradient norm by layer at depth 40, and accuracy by depth, for four configurations](https://raw.githubusercontent.com/shahzaibahmad3/ai-from-first-principles/main/depth/image.png)

> **There's a playground for this.** Open [shahzaibahmad3.github.io/ai-from-first-principles/depth/](https://shahzaibahmad3.github.io/ai-from-first-principles/depth/) (nothing to install). Every bar is one layer's gradient on a log scale. Press
> **Run** with the defaults and watch the bottom of the stack sit still. Then turn **ReLU** on.
>
> It runs a smaller version of the same experiment — 140 points instead of 240, and its own random
> number generator rather than NumPy's — because it has to do a forward and backward pass every frame.
> So its exact numbers won't match the tables below. The orders of magnitude and the orderings do,
> because those are properties of the arithmetic rather than of the sample.

---

## Where we left off

[Part 2](https://shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/) ended on a claim: the bottleneck in these models was never what they
could represent, it was what anyone could train. A nine-parameter network represents XOR exactly, and
gradient descent still fails to find that solution about a quarter of the time.

Two layers. Nine numbers. A quarter failure rate.

Now stack twenty layers and ask the same question. It stops being a probability and becomes arithmetic.

---

## The gradient does not survive the trip

Here is the only measurement that matters, and it needs no training at all — this is the network at
**initialisation**, before it has had a chance to do anything wrong. For each depth, the gradient norm
arriving at the very first weight matrix, and at the last one
([`gradient_flow.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/depth/src/gradient_flow.py)):

```
sigmoid, N(0,1) init
 depth      layer 1   last layer   attenuation    fwd RMS
     2    4.085e-02    1.577e-01      3.86e+00      0.516
     5    1.282e-02    1.089e-01      8.49e+00      0.520
    10    4.006e-04    6.467e-02      1.61e+02      0.565
    20    1.263e-06    1.046e-01      8.29e+04      0.595
    40    1.218e-10    2.088e-01      1.71e+09      0.580
```

At twenty layers the first layer receives a gradient of `0.0000013`. At forty, `0.00000000012` — one part
in 1.7 billion of what the last layer gets. Those weights are not learning slowly. They are not moving.

Look at the last column before you accept the obvious explanation. `fwd RMS` is the size of the
activations arriving at the top of the network on the way *forward*, and it is perfectly healthy at every
depth — around 0.55 whether there are 2 layers or 40. **The forward pass is fine.** Information gets all
the way up without trouble. It is only the journey back down that fails.

And the reason is embarrassingly simple. Backpropagation multiplies by the activation's derivative at
every layer it passes through, and the sigmoid's derivative is at most **0.25** — it hits that maximum
only at exactly zero, and falls away fast in both directions. So each layer keeps, at best, a quarter of
what it was handed. Twenty layers of "at best a quarter" is a factor of 4⁻²⁰, which is about one in a
trillion. There is no learning rate large enough to rescue that, and no amount of patience either.

### Where it stops learning

The nice thing about a number like `1.263e-06` is that you can go and check whether it actually matters.
Same four configurations, trained properly this time — 3,000 epochs, momentum, and *each configuration
gets its own learning-rate sweep* so nobody loses on a technicality
([`ablation.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/depth/src/ablation.py), about ten minutes):

```
points classified correctly, best of a learning-rate sweep

                                 d2      d5     d10     d20     d40
sigmoid, N(0,1)               100.0%  100.0%  100.0%   49.2%   52.1%
ReLU + He                     100.0%  100.0%  100.0%  100.0%  100.0%
ReLU + He + residual          100.0%  100.0%   50.0%   50.0%   50.0%
ReLU + He + residual + RMSNorm 100.0% 100.0%  100.0%  100.0%   62.5%
```

The sigmoid row is the point. It is perfect at ten layers and at chance by twenty — and if you put that
next to the gradient table, the cliff lands exactly where you'd predict. At depth 10 the first layer sees
`4e-04`, which is small but real. At depth 20 it sees `1.3e-06`, and the network is guessing. Two rows of
numbers, measured independently, and the accuracy falls off precisely where the gradient goes under about
one in a million.

---

## Fix one: the activation, and the scale of the weights

Two changes, both from the early 2010s, both aimed at exactly this.

**Use an activation whose derivative isn't a fraction.** ReLU is `max(0, z)`. Its derivative is exactly
**1** wherever the unit is active and 0 where it isn't. Nothing shrinks on the way through.

**Then fix the size of the initial weights.** Glorot and Bengio's 2010 scheme was derived assuming the
activation behaves roughly linearly, which is untrue of ReLU — it zeroes half its input. He et al. redid
the derivation for rectifiers in 2015 and got a factor of √2 more variance.

Together:

```
ReLU + He init
 depth      layer 1   last layer   attenuation    fwd RMS
     2    2.450e-02    6.401e-02      2.61e+00      0.513
     5    2.037e-02    5.161e-02      2.53e+00      0.364
    10    1.577e-02    2.439e-02      1.55e+00      0.105
    20    1.329e-02    1.299e-02      9.77e-01      0.058
    40    5.822e-04    2.511e-04      4.31e-01      0.011
```

At depth 20 the attenuation is **0.98** — the first layer gets essentially the same gradient as the last.
Compare that to sigmoid's 82,854× at the same depth. And the accuracy row above is 100% at every depth
tested, including 40.

That is the whole 2012 recipe, and it is worth sitting with how small it is. Not a new architecture. An
activation function and a number to multiply the initial weights by.

(One thing to notice for later: `fwd RMS` is now sliding downward with depth — 0.51 to 0.011. The gradient
is fine and the *forward* signal has started to thin. There are two directions to lose a signal in, and
we have only fixed one.)

---

## Fix two: the famous one, which makes it worse

Residual connections are the most celebrated idea in this whole story. Instead of `out = f(x)`, a block
computes `out = f(x) + x`, so the layer only has to learn a *correction* to what it was given. Gradients
get a route home that skips the weights entirely. This is the idea in every ResNet and every transformer.

So add it, change nothing else, and measure:

```
ReLU + He + residual
 depth      layer 1   last layer   attenuation    fwd RMS
     2    2.450e-02    6.401e-02      2.61e+00      0.513
     5    7.251e-02    3.313e-01      4.57e+00      2.518
    10    1.383e-03    5.712e-03      4.13e+00     18.773
    20    0.000e+00    0.000e+00          dead   4167.502
    40    0.000e+00    0.000e+00          dead  220742461.125
```

Accuracy collapses to chance from ten layers on. And at twenty, the gradient is not small — it is
**exactly zero**. Not 1e-40. Zero, in double precision, at every layer.

The `fwd RMS` column says why, and it is the same column that looked so reassuring for sigmoid. `0.5`,
then `2.5`, then `18.8`, then four thousand, then two hundred million. Adding the input back at every
block means the variance accumulates rather than passing through:

```
Var(x_next) = Var(x) + Var(f(x))
```

Every block adds. Nothing divides. So the activations grow steadily with depth, and by layer 20 they are
large enough to drive the output sigmoid completely flat. A saturated sigmoid has a derivative that
underflows to zero, and zero times anything, all the way down, is zero. The gradient doesn't vanish
gradually here — it is annihilated in one step at the top and there is nothing left to propagate.

This is not a toy artifact. The linear growth of variance through a residual stack is exactly what the
literature describes, and it is why residual blocks are never shipped bare.

---

## Fix three: normalisation, and why it comes as a pair

Put a normaliser after each block and the growth stops. I used **RMSNorm** — divide the activations by
their root-mean-square — because it is what modern LLMs actually use (Llama, Mistral, Gemma, Qwen all
replaced LayerNorm with it), and because its backward pass is four lines instead of ten:

```
ReLU + He + residual + RMSNorm
 depth      layer 1   last layer   attenuation    fwd RMS
     2    6.560e-02    2.118e-01      3.23e+00      1.000
     5    4.566e-02    2.987e-01      6.54e+00      1.000
    10    4.836e-02    2.832e-01      5.86e+00      1.000
    20    6.710e-04    2.615e-01      3.90e+02      1.000
    40    4.061e-05    2.905e-01      7.16e+03      1.000
```

`fwd RMS` is `1.000` at every depth, because that is definitionally what RMSNorm does. The gradient comes
back, and accuracy returns to 100% through twenty layers. Residual plus normalisation — that pair, in that
order — is the repeating unit of every transformer you have used.

---

## What this experiment cannot tell you

This is the part I would want to read, so here it is rather than buried.

**At this scale, fix one was already enough, and fixes two and three did not help.** Look again at the
accuracy table. `ReLU + He` scores 100% at every depth including 40. `ReLU + He + residual + RMSNorm`
matches it up to twenty layers and then does *worse* — 62.5% at depth 40. On a two-dimensional problem
with sixteen hidden units, the residual/normalisation machinery costs something and buys nothing.

That is not evidence against residuals. It is evidence that a 2D toy is too small to need them. Residual
connections earn their reputation at depths and widths this experiment cannot reach, on problems where the
extra layers have something to do. What the experiment *can* honestly show you is the mechanism — why
residuals alone blow up, and what normalisation does about it — and it shows that cleanly. Anything about
how they behave at 96 layers on real data, I am taking from the literature, not from these numbers.

**The 0.125-style precision stops at the mechanism.** The percentages here come from one 240-point
problem, one width, hand-written momentum descent, five seeds. The *ordering* is robust and the mechanisms
are real. The specific numbers are not ImageNet.

**"Internal covariate shift" is probably not why normalisation works.** That explanation comes from the
original BatchNorm paper (2015) and it is the version most of us absorbed. Santurkar et al. tested it in
2018 by deliberately *injecting* distribution shift after the normaliser — and the benefits persisted.
Their account is that it smooths the optimisation landscape instead. The technique works; the famous
story about why is not settled.

**Normalisation isn't strictly necessary — variance control is.** Fixup (2019) trains very deep residual
networks with no normalisation at all, purely by scaling the initialisation to account for depth. That is
the cleaner statement of the problem: the issue was never a missing normaliser, it was that nobody was
accounting for what the skip connections do to variance.

**I did not reproduce the degradation result.** The ResNet paper's famous observation is that a deeper
plain network can do worse on *training* error than a shallower one even though it could represent it by
learning identity in the extra layers. My sigmoid rows look like that, but the cause here is demonstrably
the vanishing gradient, which is a different claim. I am citing that result, not measuring it.

---

## The history, briefly

Backpropagation was in print in 1970 ([Part 2](https://shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/)). Depth started working around
2012. The gap is not mysterious once you have the tables above, and the people involved knew:

- **1991** — Sepp Hochreiter analyses the vanishing gradient in his diploma thesis. It is written in
  German, which is a large part of why it went unnoticed for years.
- **1994** — Bengio, Simard and Frasconi publish *Learning long-term dependencies with gradient descent
  is difficult*, and the problem has a name in English.
- **2010** — Glorot and Bengio quantify how initialisation scale interacts with depth. Nair and Hinton
  introduce ReLU into this setting; Glorot, Bordes and Bengio popularise it the following year.
- **2012** — AlexNet uses ReLU and reports reaching a 25% training error rate about **six times faster**
  than the equivalent network with tanh.
- **2015** — He et al. redo the initialisation derivation for rectifiers. BatchNorm arrives. So does the
  residual connection.
- **2016 → 2019** — LayerNorm, then RMSNorm, which is what the current generation of LLMs runs on.

Twenty years between naming the problem and having the standard fixes. Not because the mathematics was
hard, but because "multiply by 0.25, twenty times" is the kind of thing that is obvious only once someone
has plotted it.

---

## What I take from this

Part 2's lesson was that optimisation, not representation, was the wall. Part 3 is what that wall looks
like when you measure it: a number getting smaller by a factor of four per layer until it disappears.

Everything that fixed it was arithmetic on signal size. Not a cleverer architecture — an activation with
derivative 1 instead of 0.25, a √2 in the initialisation, a division by the root-mean-square. The
"deep" in deep learning is downstream of getting those three right.

And the general shape holds today. When a large model won't train, the first questions are still about
scale: how the weights were initialised, what the learning rate is doing, whether anything is saturating,
where the normalisation sits. The specifics have moved on. The failure mode has not.

---

## Run it yourself

```bash
git clone https://github.com/shahzaibahmad3/ai-from-first-principles
cd ai-from-first-principles/depth
pip install -r requirements.txt

python3 src/depth.py            # gradient check + a smoke test, instant
python3 src/gradient_flow.py    # the attenuation tables, no training needed
python3 src/ablation.py         # accuracy by config x depth -- about 10 minutes
python3 src/figure.py           # regenerates image.png from results.json

open index.html                  # the interactive version, no install
```

`depth.py` checks its own backward pass against finite differences before doing anything else. That is
not decoration: a hand-written gradient through skip connections and a normaliser is easy to get subtly
wrong, and a subtly wrong gradient still trains — just worse. Mine was wrong the first time, and the
check is what caught it.

`ablation.py` writes `results.json`; `figure.py` reads it, so the figure and the tables above cannot
disagree.

## Sources

- Hochreiter, S. (1991). *Untersuchungen zu dynamischen neuronalen Netzen* — diploma thesis; see [Schmidhuber's annotated history](https://people.idsia.ch/~juergen/deep-learning-history.html) for its place in the record
- [Bengio, Simard & Frasconi (1994), *Learning long-term dependencies with gradient descent is difficult*](https://www.comp.hkbu.edu.hk/~markus/teaching/comp7650/tnn-94-gradient.pdf)
- [Glorot & Bengio (2010), *Understanding the difficulty of training deep feedforward neural networks*](https://proceedings.mlr.press/v9/glorot10a.html)
- [Nair & Hinton (2010), *Rectified Linear Units Improve Restricted Boltzmann Machines*](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) and [Glorot, Bordes & Bengio (2011), *Deep Sparse Rectifier Neural Networks*](https://proceedings.mlr.press/v15/glorot11a.html)
- [Krizhevsky, Sutskever & Hinton (2012), *ImageNet Classification with Deep Convolutional Neural Networks*](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
- [He, Zhang, Ren & Sun (2015), *Delving Deep into Rectifiers*](https://arxiv.org/abs/1502.01852)
- [Ioffe & Szegedy (2015), *Batch Normalization*](https://arxiv.org/abs/1502.03167) and [Santurkar et al. (2018), *How Does Batch Normalization Help Optimization? (No, It Is Not About Internal Covariate Shift)*](https://arxiv.org/abs/1805.11604)
- [He et al. (2015), *Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385)
- [De & Smith (2020), *Batch Normalization Biases Residual Blocks Towards the Identity Function*](https://arxiv.org/abs/2002.10444) — on how variance grows through a residual stack
- [Zhang, Dauphin & Ma (2019), *Fixup Initialization: Residual Learning Without Normalization*](https://arxiv.org/abs/1901.09321)
- [Ba, Kiros & Hinton (2016), *Layer Normalization*](https://arxiv.org/abs/1607.06450) and [Zhang & Sennrich (2019), *Root Mean Square Layer Normalization*](https://arxiv.org/abs/1910.07467)

---

**Next → attention.** Everything so far has been one unit, stacked. The mechanism that made these models
into something you can talk to is a different idea altogether, and it is the last piece before the whole
picture closes.

**Play with it yourself:** [shahzaibahmad3.github.io/ai-from-first-principles/depth/](https://shahzaibahmad3.github.io/ai-from-first-principles/depth/)
**Full code:** [github.com/shahzaibahmad3/ai-from-first-principles](https://github.com/shahzaibahmad3/ai-from-first-principles)

*This is Part 3 of "AI, from first principles" — I use AI heavily, every day, for real work. This series
is me making the machinery underneath it concrete, one genuinely-understood idea at a time, with
something you can run for every claim.*
