# XOR Was Never the Hard Part. Finding the Weights Was.

### Part 2 of *AI, from first principles*: the fix for the wall one neuron hit — and why the fix turned out to be the easy half.

![60 training runs, identical architecture, different starting weights](https://raw.githubusercontent.com/shahzaibahmad3/ai-from-first-principles/main/finding-the-weights/image.png)

> **There's a playground for this.** Open it at [shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/](https://shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/) — nothing to
> install. Press **Run** and watch a two-unit hidden layer bend a flat surface into XOR. Then press
> **New random init** a few times and watch it fail. Everything below explains what you're seeing.

---

## Where we left off

[Part 1](https://shahzaibahmad3.github.io/ai-from-first-principles/neuron/) ended at a wall. One neuron computes a weighted sum and fires if it crosses
a threshold, which is geometrically a straight line — and no straight line separates XOR. Load XOR in
that playground and the line lurches back and forth forever.

The fix is the obvious one: use more than one neuron. Put two of them in a **hidden layer**, let a third
neuron combine their outputs, and the boundary is no longer a single line — it's a region carved out by
two lines at once. It can bend. Open this post's playground on XOR and you can watch it happen: a flat,
undecided surface folds into a diagonal band with the two "fire" corners on the outside.

So that's the answer, the wall comes down, and the post could end here.

Except that the interesting thing about this story is that **the part I just described was never what
held the field up.**

---

## The nine numbers

Here is a network that computes XOR exactly. It has two hidden units and nine parameters, and it did no
training at all — I wrote the numbers down by hand:

```python
W1 = [[1, 1],
      [1, 1]]      # both hidden units see both inputs
b1 = [-0.5, -1.5]  # unit 1 fires if a+b >= 1 (OR); unit 2 if a+b >= 2 (AND)
W2 = [1.0, -2.0]   # take OR, then veto with twice the AND
b2 = -0.5
```

That's it. The trick is one line of boolean algebra:

```
XOR(a, b) = OR(a, b) AND NOT AND(a, b)
```

"Either one of them, but not both." The first hidden unit computes OR, the second computes AND, and the
output unit adds the OR and subtracts twice the AND, so the both-on case gets pushed back under the
threshold. Run [`capacity.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/finding-the-weights/src/capacity.py) and it prints the truth table and asserts the result:

```
 a  b |  OR  AND |  out  want
------------------------------
 0  0 |   0    0 |    0     0
 0  1 |   1    0 |    1     1
 1  0 |   1    0 |    1     1
 1  1 |   1    1 |    0     0
```

You can do this in the playground too — **Solve it by hand instead** loads exactly these weights, and the
network is correct at epoch zero.

This was never a secret. Rosenblatt's *Principles of Neurodynamics* (1962) devotes the whole of Part III
— chapters 15 through 20 — to multi-layer and cross-coupled perceptrons. Multilayer networks were not an
idea anyone was missing.

**What Rosenblatt could not do was train them.**

---

## The actual problem: nobody knows what a hidden unit should have said

Part 1's learning rule was one line of arithmetic:

```
error = target − prediction
```

Look at what that needs. It needs a **target** — a statement of what this unit was supposed to output. For
the output neuron you have one: it's the label in your training data.

For a hidden unit, there is nothing. Your data says XOR(0,1) = 1. It says nothing whatsoever about what
the second unit in the middle layer ought to have produced along the way. No target, no error, no update.

That is the **credit assignment** problem: when the network as a whole is wrong, how much of the blame
belongs to each unit buried inside it? And it — not representational capacity — is the thing that
actually cost the field a decade and a half.

The answer is backpropagation, and the load-bearing line is short enough to read. From
[`credit_assignment.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/finding-the-weights/src/credit_assignment.py):

```python
delta_out = (a2 - target) * a2 * (1 - a2)          # error at the output
grad_W2   = a1.T @ delta_out / n

# the credit-assignment step: project the output error back through W2
delta_hidden = (delta_out @ self.W2.T) * a1 * (1 - a1)
grad_W1      = X.T @ delta_hidden / n
```

The middle line is the whole idea. A hidden unit's share of the blame is the output error, multiplied by
the weight through which that unit influenced the output. A unit connected by a large weight gets a large
share; one connected by a near-zero weight gets almost none. You never needed a target for the hidden
layer — you needed a way to *distribute* the error you already had. In the playground, press **Step** and
the diagram pulses backwards along that path: output first, then the weights it flows through, then the
hidden units.

---

## So now we can train it. Can we?

Here is the experiment I actually wanted to run. Take the network whose exact solution I wrote out by
hand above. Start it from random weights. Train it with backpropagation for 20,000 epochs. Do that 300
times from 300 different random starts, at four different hidden-layer widths — 1,200 full training runs,
about eight minutes in [`credit_assignment.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/finding-the-weights/src/credit_assignment.py):

```
 hidden    solved   failed   median final loss
----------------------------------------------
      2    74.3%       77              0.0003
      3    98.3%        5              0.0003
      4    99.3%        2              0.0002
      8   100.0%        0              0.0002
```

**A quarter of the runs never solve XOR.** Same nine-parameter architecture whose exact solution fits on
one screen, same learning rate, same number of epochs. The only difference is where the weights started.

The spread among the *successful* runs is striking on its own. Measuring when each first drops below 0.01
loss, the fastest arrives at epoch 1,280 and the slowest takes 8,760 — nearly 7× longer, same everything.
The left panel of the figure up top is 60 of those runs on a log axis; the blue threads are the ones that
get there, at wildly different times.

### What failure actually looks like

The failures aren't noise. They have a signature, and it's oddly human. Thirteen of those 60 traced runs
never solved it; [`inspect_stuck.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/finding-the-weights/src/inspect_stuck.py) retrains them and prints what they
actually output. Here is the whole list, unabridged:

```
13 of 60 runs failed to solve XOR

 seed  final loss   outputs
----------------------------------------------------------
    5      0.1673   [0.335  0.977  0.332  0.334]
    6      0.1254   [0.020  0.499  0.984  0.500]
   13      0.1254   [0.019  0.983  0.500  0.501]
   18      0.1254   [0.017  0.984  0.499  0.500]
   21      0.1254   [0.018  0.500  0.982  0.501]
   27      0.1254   [0.015  0.982  0.500  0.501]
   28      0.1254   [0.017  0.499  0.983  0.501]
   34      0.1254   [0.014  0.500  0.981  0.501]
   40      0.1254   [0.014  0.500  0.977  0.501]
   50      0.1671   [0.020  0.666  0.666  0.667]
   53      0.1255   [0.015  0.500  0.978  0.501]
   54      0.1254   [0.016  0.500  0.984  0.501]
   56      0.1254   [0.016  0.500  0.983  0.501]

median final loss among the failures: 0.12541
```

Eleven of the thirteen do the same thing. They learn two of the four points confidently — 0.02 where the
answer is 0, 0.98 where it's 1 — and on the remaining two they output **exactly 0.5**. A coin flip. The
network has learned half of XOR and is hedging on the rest, and the arithmetic of that hedge is exact: a
point sitting at 0.5 contributes 0.25 to the squared error, so hedging on two of four costs
`(0 + 0 + 0.25 + 0.25) / 4 = 0.125`. That's the plateau in the figure.

It sits there because it's stuck in the strict sense: improving either remaining point requires first
making the other worse, and gradient descent won't take a step uphill to get there.

The other two fail a different way. Seed 50 learns one point and then emits a single constant, 0.667, for
the other three — which is precisely the average of those three targets, the best a single number can do
when you've stopped distinguishing between them. Worth being exact about this rather than rounding it
into the tidier story: most failures hedge at 0.125, and a couple collapse instead, at 0.167. The
playground names whichever one you've landed on.

### The fix, and what it tells you

Look at what repairs it. Going from 2 hidden units to 8 takes the success rate from 74% to 100%.

**That extra width adds no representational power whatsoever.** Two units already represent XOR exactly —
provably, by construction, nine numbers, printed above. What the extra units add is more directions to
descend in, so optimisation gets trapped less often.

This is an optimisation story from beginning to end. And it isn't a historical curiosity: "make it wider
and it trains more reliably" is durable practical folklore in deep learning, and a nine-parameter network
is the smallest place I know of to watch the mechanism behind it.

> Try it in the playground: **Run 100 inits × 4 widths** reruns that whole table in your browser — 400
> trainings, a few seconds, filling in a width at a time. It uses its own random number generator rather
> than NumPy's, so the exact per-seed values differ — but it lands on the same rates, because the rates
> are a property of the problem, not of the generator. Switch to **Rings** for the contrast: there, width
> buys genuine capacity, because two hidden units give you two lines, and no two lines enclose a blob.

---

## While we're here: the story around this is also wrong

The version most of us absorbed goes something like:

> In 1969 Minsky and Papert proved neural networks couldn't compute XOR. The field lost faith, funding
> collapsed, the first AI winter began, and backpropagation in 1986 undid the damage.

I've repeated a version of that myself. Having gone and read what the book argues and when the money
actually moved, I think it's wrong in three separate places.

**The theorems aren't about XOR, and they're correct.** *Perceptrons: An Introduction to Computational
Geometry* is a book about the **order** of a predicate — the minimum number of inputs a single unit must
look at simultaneously to compute it. AND, OR and MAJORITY are order 1, and those turn out to be exactly
the linearly separable functions. **Parity** is not of finite order: for N inputs it needs a unit wired to
every input at once. XOR is just parity at N = 2 — the case that fits on a slide. The result they clearly
cared more about is **connectedness** — is this shape one piece or two? — which requires order Ω(√N),
and the intuition is genuinely lovely: to decide whether a figure is connected you have to *see across*
it, and no committee of units each peering at a small local patch can answer that, however many you use.
These are sharp, true statements about what one restricted architecture can represent. They are not the
claim that neural networks cannot learn.

**The winter's causality is oversold.** Larger forces were moving than one book about perceptrons:

- **ALPAC (1966)** concluded machine translation was slower, costlier and worse than human translation,
  and ended major US funding for it — **three years before** *Perceptrons* was published.
- **The Lighthill Report (1973)**, for the UK Science Research Council, attacked AI for failing its
  grandiose objectives, with combinatorial explosion as the central charge — not perceptrons. UK funding
  collapsed almost immediately after.
- **The Mansfield Amendment** restricted US defence research funding to work with direct military
  relevance, and general AI research lost its patron. (A detail worth getting right, because it's
  commonly garbled: ARPA was renamed DARPA in **1972**, which was a separate event, not a consequence of
  the funding restriction.)
- Thomas Haigh has argued in *Communications of the ACM* (December 2023) that
  [there was no first AI winter at all](https://cacm.acm.org/opinion/there-was-no-first-ai-winter/) —
  that measured by researchers, publications, students and conference attendance, the 1970s look less
  like a bust than like a field professionalising. SIGART membership roughly doubled between 1969 and
  1973, then nearly tripled again by 1978.

**And 1986 wasn't a rescue by invention.** The timeline undercuts the tidy ending:

- **1970** — **Seppo Linnainmaa** publishes reverse-mode automatic differentiation, the algorithm itself, with FORTRAN code, in his MSc thesis (journal version, *BIT*, 1976).
- **1974** — **Paul Werbos**'s PhD thesis *Beyond Regression* discusses applying it to neural networks.
- **1982** — Werbos publishes an explicit neural-network application.
- **1986** — **Rumelhart, Hinton & Williams** show it learns useful representations in hidden layers, and the field finally pays attention.

The 1974-versus-1982 line is genuinely contested, and I'd rather flag that than pick a winner:
Schmidhuber's priority history argues the first neural-network application is the 1982 paper and "not yet
in his 1974 thesis, as is sometimes claimed." A post about a misremembered history should be careful
about its own dates.

Either way the shape holds. The mathematics needed to train hidden layers was in print by 1970 and sat
largely unused for over a decade. Making an idea influential is a real contribution — but the 1986
paper's achievement was demonstration and persuasion, not invention.

---

## What I take from this

The bottleneck was never what these models could **represent**. It was what anyone could **train**.
Minsky and Papert's theorems, read correctly, describe the representational limits of one restricted
architecture — and the field's actual paralysis was somewhere else entirely, in a credit-assignment
problem whose solution was published in 1970 and largely ignored.

That distinction hasn't aged out. Most of the distance between a model that works and one that doesn't
is still optimisation, initialisation and data, not architecture. A nine-parameter network that fails a
quarter of the time on a problem you can solve by hand in one line of boolean algebra is a decent
reminder of how old that lesson is.

---

## Run it yourself

```bash
git clone https://github.com/shahzaibahmad3/ai-from-first-principles
cd ai-from-first-principles/finding-the-weights
pip install -r requirements.txt

python3 src/capacity.py            # hand-built XOR, no training, instant
python3 src/credit_assignment.py   # 1,200 training runs -- about 8 minutes
python3 src/inspect_stuck.py       # what the failures actually output
python3 src/figure.py              # regenerates image.png from results.json

open index.html                    # the interactive version, no install
```

`credit_assignment.py` writes `results.json`; `figure.py` reads it, so the figure and the table can
never disagree. Every number quoted above comes out of one of these scripts.

## Sources

- [Minsky & Papert, *Perceptrons* — Léon Bottou's review of the 2017 reissue](https://leon.bottou.org/publications/pdf/perceptrons-2017.pdf)
- [*Perceptrons* (book) — overview of the results](https://en.wikipedia.org/wiki/Perceptrons_(book)) (order, parity as Theorem 3.1.1, connectedness as Ω(√N))
- [Rosenblatt, *Principles of Neurodynamics* (1962) — full text](https://gwern.net/doc/ai/nn/1962-rosenblatt-principlesofneurodynamics.pdf) (Part III, chapters 15–20, is the multi-layer material)
- [Who Invented Backpropagation? — Jürgen Schmidhuber](https://people.idsia.ch/~juergen/who-invented-backpropagation.html)
- [Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*](https://www.nature.com/articles/323533a0)
- [Haigh, *There Was No 'First AI Winter'* — Communications of the ACM, Dec 2023](https://cacm.acm.org/opinion/there-was-no-first-ai-winter/)
- [The Lighthill Report and UK AI funding — *A brief history of AI* (arXiv:2109.01517)](https://arxiv.org/pdf/2109.01517)

---

**Next → the part where stacking these stops working, and what had to be invented before depth paid off.**

**Play with it yourself:** [shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/](https://shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/)
**Full code:** [github.com/shahzaibahmad3/ai-from-first-principles](https://github.com/shahzaibahmad3/ai-from-first-principles)

*This is Part 2 of "AI, from first principles" — I use AI heavily, every day, for real work. This series
is me making the machinery underneath it concrete, one genuinely-understood idea at a time, with
something you can run for every claim.*
