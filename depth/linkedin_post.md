Backpropagation was published in 1970. Deep learning started working around 2012.

I filed that gap under "we needed GPUs and data". Then I measured it, and the real answer is smaller and stranger.

Take a network 20 layers deep and look at the gradient arriving at each layer — before any training, just at initialisation:

    depth      layer 1    last layer
        2    4.09e-02      1.58e-01
       10    4.01e-04      6.47e-02
       20    1.26e-06      1.05e-01
       40    1.22e-10      2.09e-01

At 40 layers the first layer receives one part in 1.7 BILLION of what the last layer gets. Those weights aren't learning slowly. They aren't moving.

Here's the part that surprised me: the forward pass is completely fine. Activations arrive at the top at healthy magnitude whether there are 2 layers or 40. Only the trip back down fails.

And the reason is almost insultingly simple. Backprop multiplies by the activation's derivative at every layer, and a sigmoid's derivative maxes out at 0.25. Each layer keeps at most a quarter. Twenty layers of that is 4^-20 — about one in a trillion. No learning rate saves you from that.

Does it matter? Trained properly, each config with its own learning-rate sweep: sigmoid is 100% correct at 10 layers, and at chance by 20.

The cliff lands exactly where the gradient crosses one-in-a-million. Two independent measurements, same threshold.

Then the fixes, which are shockingly small.

ReLU's derivative is exactly 1 where it's active, not 0.25. He initialisation scales the starting weights for an activation that zeroes half its input. That's it — an activation function and a √2. Attenuation at depth 20 goes from ~83,000× to 0.98×, and accuracy hits 100% at every depth I tested.

Not a new architecture. Arithmetic on signal size.

Then I added residual connections — the most celebrated idea in the whole story — and it got WORSE. Accuracy collapsed to chance from 10 layers on, and at 20 layers the gradient was exactly zero. Not small. Zero, in double precision.

A skip connection adds its input back, so variance accumulates: Var(next) = Var(x) + Var(f(x)). Every block adds, nothing divides. Forward activations went 0.5 → 2.5 → 18.8 → 4,000 → 200 million, until the output saturated flat and its derivative underflowed. The gradient wasn't attenuated — it was annihilated in one step.

Add normalisation and it comes back. Residual + normalisation, as a pair — that's the repeating unit of every transformer you've used.

One honest caveat, because it's the interesting part: at this scale ReLU + He was already enough, and residual + norm bought nothing — it did worse at 40 layers. A 2D toy is too small to need them. What it does show cleanly is the mechanism.

Playground (watch the bars die layer by layer, then flip ReLU on), write-up and code — gradient-checked, every number reproducible: [link in comments]

Next: attention.

#DeepLearning #MachineLearning #NeuralNetworks #AI #Transformers
