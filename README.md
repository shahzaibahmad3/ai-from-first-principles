# AI, From First Principles

A series on the actual mechanics of the models we use — working from the perceptron up, one
genuinely-understood idea at a time. Every post ships with a runnable playground and the plain
code behind it, no libraries, no magic.

**Live site:** https://shahzaibahmad3.github.io/ai-from-first-principles/

## Posts

| # | Post | Code |
|---|------|------|
| 1 | [The Neuron — the unit an LLM is built from](https://shahzaibahmad3.github.io/ai-from-first-principles/neuron/) | [`neuron/`](neuron/) |
| 2 | [XOR Was Never the Hard Part. Finding the Weights Was.](https://shahzaibahmad3.github.io/ai-from-first-principles/finding-the-weights/) | [`finding-the-weights/`](finding-the-weights/) |
| 3 | [Stacking Was the Easy Part. Getting the Gradient Back Down Wasn't.](https://shahzaibahmad3.github.io/ai-from-first-principles/depth/) | [`depth/`](depth/) |

More parts land here as the series continues.

## Running a post locally

Each post folder is self-contained:

```bash
cd finding-the-weights
open index.html                    # the playground, no install needed
pip install -r requirements.txt
python3 src/capacity.py            # hand-built XOR, no training
python3 src/credit_assignment.py   # the 1,200-run experiment
```
