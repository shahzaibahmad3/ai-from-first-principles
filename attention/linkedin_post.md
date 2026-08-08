Attention has no idea what order your words are in. I don't mean approximately — I measured it.

Run some tokens through an attention head and note the output. Shuffle the tokens and run them again. Over 200 random trials the largest difference was 2.9e-15.

Machine epsilon for a 64-bit float is 2.2e-16. That's about thirteen rounding errors — the noise from a few matrix multiplications. Shuffling the input shuffles the output rows and changes not one meaningful bit.

Look at the formula and it's obvious:

    A   = softmax(Q Kᵀ / √d)
    out = A V

Every token projects a query (what am I looking for), a key (what do I advertise), and a value (what I hand over if you listen). Compare every query to every key, softmax, take the weighted average.

There is no index in there anywhere. Q Kᵀ compares token i with token j using only what those two tokens CONTAIN — never where either sits. It's a soft dictionary lookup over an unordered bag.

So I trained one head to do exactly that: items each carrying a key and a value, then a query carrying only a key. Return the matching value. No positional encoding, no mask, no feed-forward layer, plain SGD, backprop by hand.

100% accuracy on all three seeds — and its attention weight lands on the matching key 100% of the time.

Precisely, because "it spikes on the match" overstates it — over 400 sequences the query puts 0.48 on the matching item and 0.46 on itself (attending to yourself is normal; its own value is a constant the readout subtracts off). The number that matters: 0.02 on each WRONG item. A 25× contrast among the candidates. That's a lookup, not a guess.

Then where does word order come from? It's added. Sinusoids in the 2017 paper; RoPE in every current open model. Order is a separate signal bolted onto the input, not a property of the mechanism.

And there's a second source people forget: the causal mask. If a token may only look backwards, which tokens it can see IS positional information. Measured, it breaks order-blindness just as hard — 8.2, with no positional encoding involved.

One more thing. "Attention Is All You Need" is about what attention REPLACED — recurrence and convolution. Attention itself is Bahdanau et al., 2014, three years earlier.

And attention isn't where the computation lives. Per layer: attention is 4d², the feed-forward sublayer is 8d². Exactly 2:1, for any width. GPT-3: 58B parameters of attention, 116B of feed-forward.

Those feed-forward layers hold 4,718,592 neurons — exactly the "about five million" I quoted in part 1. Four posts in, the umbrella decision is still the unit, and still where most of the model lives.

Attention decides which tokens get to talk. The neurons do the thinking about what they said.

Playground (shuffle the tokens yourself, then flip positional encoding on), write-up and code — every number reproducible: [link in comments]

Next: tokenisation.

#MachineLearning #DeepLearning #Transformers #AI #LLM
