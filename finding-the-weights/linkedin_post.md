Last post ended at a wall: one neuron draws one straight line, and no straight line separates XOR.

The fix is the obvious one — use more than one. Two neurons in a hidden layer, a third to combine them, and the boundary can bend.

What I didn't expect, when I went and measured it, is that the fix was never the hard part.

You can write XOR down by hand. Nine numbers, no training:

    XOR(a, b) = OR(a, b) AND NOT AND(a, b)

One hidden unit computes OR, the other AND, the output takes the OR and subtracts twice the AND. Done. And this was never a secret — Rosenblatt's 1962 book gives its entire Part III to multi-layer networks.

What he couldn't do was TRAIN them.

The learning rule from last post needs one thing: error = target − prediction. For the output neuron, fine — the target is your label. For a neuron buried in the middle, there's nothing. Your data says XOR(0,1) = 1. It says nothing about what the second hidden unit should have output on the way there. No target, no error, no update.

That's the credit assignment problem, and it — not capacity — is what cost the field a decade and a half.

So I ran it. Same 9-parameter network whose solution I just wrote down. 300 random starts, 20,000 epochs, four widths — 1,200 training runs:

→ 2 hidden units: 74.3% solve XOR. 25.7% never do.
→ 8 hidden units: 100%.

A quarter of runs fail at a problem I solved by hand in one line of boolean algebra. Same architecture, same learning rate. Different starting weights.

And the failure mode is oddly human. I traced 60 of those runs; 13 never got there. Eleven of the 13 do the same thing: learn two of the four points confidently, then output exactly 0.5 on the other two. A coin flip. They've learned half of XOR and are hedging on the rest — because improving either remaining point requires first getting worse, and gradient descent won't step uphill to get there. That hedge costs exactly 0.125 loss, and you can watch them sit on it.

(The other two fail differently, collapsing to one constant across the points they stopped distinguishing.)

Now: going from 2 units to 8 fixes it. And that extra width adds ZERO representational power — two units already represent XOR exactly, provably, nine numbers. What width adds is more directions to descend in, so optimisation gets trapped less often.

"Make it wider and it trains more reliably" is durable deep-learning folklore. A 9-parameter network is the smallest place I know of to watch the mechanism behind it.

The bottleneck was never what these models could represent. It was what anyone could train. Still true — most of the distance between a model that works and one that doesn't is optimisation, initialisation and data, not architecture.

Playground (re-roll the init until it fails), write-up and code — every number reproducible: [link in comments]

Next: where stacking these stops working.

#MachineLearning #DeepLearning #NeuralNetworks #AI #Backpropagation
