# Attention Has No Idea What Order Your Words Are In.

### Part 4 of *AI, from first principles*: the mechanism everyone name-drops, measured — including the part of it that turns out to be missing.

![What one head learns, and the order test](https://raw.githubusercontent.com/shahzaibahmad3/ai-from-first-principles/main/attention/image.png)

> **There's a playground for this.** Open [shahzaibahmad3.github.io/ai-from-first-principles/attention/](https://shahzaibahmad3.github.io/ai-from-first-principles/attention/) (nothing to install). Press **Shuffle the tokens** and watch the number stay at
> zero. Then turn on positional encoding, or the causal mask, and watch it stop being zero.

---

## Where we left off

Parts [1](https://shahzaibahmad3.github.io/ai-from-first-principles/neuron/), [2](https://shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/) and [3](https://shahzaibahmad3.github.io/ai-from-first-principles/depth/)
were all about one unit, stacked: what a neuron does, how you find its weights, and what goes wrong when
you pile the layers up. Attention is the first genuinely different idea in this series.

It is also the one with the most mystique attached, so let me start with the thing that surprised me most
when I went and measured it.

---

## Shuffle the words and the answer doesn't change

Take a handful of tokens, run them through one attention head, and note the output. Now shuffle the
tokens and run them through again. If attention understood word order at all, you would get a different
answer.

You get the same answer. Not approximately — measurably, exactly the same. Over 200 random inputs, random
weights and random shufflings ([`permutation.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/attention/src/permutation.py)):

```
shuffling the input tokens, worst case over 200 random trials

  configuration              max output shift   max attn shift
  ------------------------------------------------------------
  plain attention                   2.887e-15        5.551e-16
  + positional encoding             7.716e+00        9.855e-01
  + causal mask                     8.206e+00        1.000e+00
```

Machine epsilon for a 64-bit float is `2.220e-16`. The worst deviation over two hundred trials is about
thirteen of those — which is what you get from rounding error accumulating through a few matrix
multiplications, and nothing else. Shuffling the input shuffles the output rows and changes not one
meaningful bit.

The attention matrix tells the same story even more plainly: shuffle the tokens and you get the *same
matrix*, with its rows and columns permuted to match. Same numbers, moved.

---

## Why — there is no position in the formula

Here is the whole mechanism:

```
A   = softmax(Q Kᵀ / √d)        who to listen to
out = A V                       listen to them
```

Every token projects itself three ways. **Q** (query) is what it is looking for. **K** (key) is what it
advertises about itself. **V** (value) is what it hands over if you do listen to it. Compare every query
against every key, turn those similarities into weights that sum to one, and take the weighted average of
the values.

Read that again and look for an index. There isn't one. `Q Kᵀ` compares token *i* with token *j* using
only what those two tokens contain — never where either of them sits. It is a **soft dictionary lookup**
over an unordered bag, and being blind to order isn't a flaw in the implementation; it is what the
formula says.

The √d is worth a sentence, since it looks arbitrary. Dot products of *d*-dimensional vectors grow like
√d, and a softmax over large numbers saturates into a hard argmax with almost no gradient — exactly the
dead end from [Part 3](https://shahzaibahmad3.github.io/ai-from-first-principles/depth/). Dividing by √d keeps the scores in the range
where softmax still has a usable slope. It is the same "get the signal size right" move, again.

---

## So watch one do the lookup

If attention is a dictionary lookup, a single head should be able to *learn* to look something up. So
here is the smallest task that asks exactly that: a few items, each carrying a key and a value, then a
query carrying only a key. Return the value that goes with the matching key.

One head. No positional encoding, no mask, no feed-forward layer, plain SGD, hand-written backprop
([`attention.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/attention/src/attention.py)):

```
 seed   accuracy   points at the matching key   final loss
  -----------------------------------------------------------
     0     100.0%                       100.0%       0.0003
     1     100.0%                       100.0%       0.0003
     2     100.0%                       100.0%       0.0003
```

The second column is the one I care about. It isn't just getting the right answer — its *attention weight
lands on the matching key*, every time, on every seed. The mechanism is doing the thing the formula
claims, and you can read it off a number.

It's worth being precise about how that weight is distributed, because "it spikes on the match" would be
a slight overstatement. Averaged over 400 sequences, the query puts **0.48** on the matching item and
**0.46** on itself — attending to yourself is normal, and its own value is a constant the readout can
subtract off. What matters is the third number: only **0.02** on each *wrong* item. Among the candidates
it is a **25× contrast**, which is a lookup, not a guess.

`attention.py` gradient-checks its own backward pass against finite differences before it does anything
else, in all four configurations. Part 3 taught me that lesson: a hand-written gradient through a softmax
is easy to get subtly wrong, and a subtly wrong gradient still trains, just worse.

---

## Then where does word order come from?

It gets added. Literally added, to the input, before attention ever sees it.

The original transformer used **sinusoids** — a fixed pattern of sines and cosines at different
frequencies, one vector per position, summed onto the token embedding. Two orderings become two different
inputs, and the mechanism that cannot see position doesn't need to, because position is now part of what
each token *is*. Current open models use **RoPE** instead (Su et al. 2021, and the default in Llama and
everything that followed it), which rotates Q and K by an angle that depends on position — a better
answer to the same problem.

Flip the toggle in the playground and the shuffle number goes from `4e-16` to about `8`.

**And there is a second source of order that is easy to forget.** A causal mask — each token may look at
itself and earlier tokens, nothing later — is *also* positional information, because which tokens you are
allowed to see depends entirely on where you are. Measured, it breaks order-blindness just as decisively
as positional encoding does: `8.206e+00` in the table above, with no positional encoding involved at all.
So "positional encoding is the only thing that gives a transformer word order" is a tidy sentence and it
is wrong. There are two.

---

## The other thing the famous title gets read as

"Attention Is All You Need" is a title about what attention **replaced**. Before 2017, sequence models
were recurrent — read one token, update a hidden state, read the next — and that is inherently serial,
which is death on a GPU. The paper's contribution was showing you could throw out recurrence and
convolution entirely and still win. Attention itself was not new: it is Bahdanau, Cho and Bengio, 2014,
three years earlier, invented to give a translation decoder a way to look back at the source sentence.

What the title is *read* as, quite often, is that attention is where the computation happens. It isn't,
and you can settle it with arithmetic. Per layer, with model width d:

```
attention      Wq, Wk, Wv, Wo   4 x d x d   =  4d²
feed-forward   d -> 4d -> d     2 x 4 x d²  =  8d²
ratio                                          2 : 1, for any d
```

The feed-forward sublayer — the plain stack of neurons from Part 1 — holds twice what attention holds,
and that ratio doesn't depend on the width at all. For GPT-3, the model Part 1 quoted
([`budget.py`](https://github.com/shahzaibahmad3/ai-from-first-principles/blob/main/attention/src/budget.py)):

```
model                  attention  feed-forward  FFN share    ratio
--------------------------------------------------------------------
GPT-3 175B                 58.0B        116.0B      66.7%    2.00:1
GPT-2 1.5B                  0.5B          1.0B      66.7%    2.00:1
Llama-2 7B                  2.1B          4.3B      66.8%    2.02:1 (gated)
Llama-2 70B (GQA)          12.1B         56.4B      82.4%    4.67:1 (gated)
```

58 billion parameters of attention and 116 billion of feed-forward. And those feed-forward layers are
where the *neurons* are: `4d × layers` for GPT-3 is **4,718,592** — which is exactly the "about five
million neurons" Part 1 opened with. Four posts later, the umbrella decision is still the unit, and it is
still where most of the model lives.

Attention is the part that decides **which** tokens get to talk to each other. The neurons are what
actually does the thinking about what they said.

---

## What this doesn't show

- **The mechanism is order-blind. A real transformer is not.** By construction — it has a positional
  signal, or a mask, or both. Nobody shipped an order-blind language model by accident. The claim here is
  about what the machinery does on its own, which is worth knowing precisely because everything else is
  built to compensate for it.
- **The retrieval task is a toy.** One head, four items, a key and a value. Real heads do much stranger
  and more interesting things — copying, tracking syntax, induction — and are far harder to interpret.
  "This is the mechanism" is the claim; "this is what a head in GPT-4 is doing" is not.
- **Parameter count is not where the time goes.** Attention is a third of the parameters, but its cost
  grows with the *square* of the sequence length while the feed-forward cost grows linearly. At long
  context, attention can dominate the compute while still being the smaller part of the model. Two
  different questions, and the parameter split answers only one.
- **The 2:1 ratio is architecture-specific**, even though it happens to be robust. It assumes the classic
  d→4d→d feed-forward. Gated SwiGLU layers use three matrices but are usually sized to land in the same
  place (Llama-2 7B comes out at 2.02:1). Grouped-query attention shrinks the key and value projections
  and pushes it much further — 4.67:1 for Llama-2 70B. Every modern change moves in the same direction,
  so "a third" is the generous reading for attention.

---

## Run it yourself

```bash
git clone https://github.com/shahzaibahmad3/ai-from-first-principles
cd ai-from-first-principles/attention
pip install -r requirements.txt

python3 src/attention.py      # gradient check, then one head learning the lookup
python3 src/permutation.py    # the order test: plain, with positions, with a mask
python3 src/budget.py         # where the parameters actually live
python3 src/figure.py         # regenerates results.json and image.png

open index.html               # the interactive version, no install
```

`figure.py` writes `results.json`, and every number quoted above is read back out of it, so the prose
and the picture cannot drift apart.

## Sources

- [Bahdanau, Cho & Bengio (2014), *Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473) — attention, three years before the transformer
- [Vaswani et al. (2017), *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) ([NeurIPS PDF](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf)) — note the position-wise feed-forward sublayer in every block
- [Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) — RoPE, the current default
- [Lee et al. (2019), *Set Transformer*](http://proceedings.mlr.press/v97/lee19d/lee19d.pdf) — attention as an operation on sets, permutation equivariance made explicit
- [Brown et al. (2020), *Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) — the GPT-3 architecture the parameter counts come from
- [You could have designed state of the art positional encoding](https://huggingface.co/blog/designing-positional-encoding) — the clearest walk from "attention has no order" to RoPE

---

**Next → tokenisation.** Every post so far has quietly assumed the model receives meaningful units. It
doesn't. It receives integers from a vocabulary somebody built, and a surprising amount of what looks
like stupidity in a language model starts there.

**Play with it yourself:** [shahzaibahmad3.github.io/ai-from-first-principles/attention/](https://shahzaibahmad3.github.io/ai-from-first-principles/attention/)
**Full code:** [github.com/shahzaibahmad3/ai-from-first-principles](https://github.com/shahzaibahmad3/ai-from-first-principles)

*This is Part 4 of "AI, from first principles" — I use AI heavily, every day, for real work. This series
is me making the machinery underneath it concrete, one genuinely-understood idea at a time, with
something you can run for every claim.*
