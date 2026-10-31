## Lesson 1 — predict the next token

In P00 you gave a model a source sequence and asked its decoder to produce a separate target. For this project, keep only a text prefix. If the prefix is `the cat`, the task is to give each possible next token a probability. There is no second source sequence and no encoder memory. The prefix itself is the context.

Start here with a pencil or Python's standard library. You do not need to create a repository today. Your first stopping point is explaining the three training rows below and calculating one loss. The rest of this course turns that calculation into a complete small language model.

### Turn one string into several questions

Use UTF-8 byte IDs for the first model. The text `hi` has bytes 104 and 105. Our inherited structural IDs are BOS=257, EOS=258 and PAD=256. BOS means “the document begins”; EOS is something the model must learn to predict. PAD is only storage filler.

| Context available at this position | Token to predict |
| --- | --- |
| BOS | `h` = 104 |
| BOS, `h` | `i` = 105 |
| BOS, `h`, `i` | EOS = 258 |

The model receives `x=[257,104,105]` and labels `y=[104,105,258]`. This is a one-position shift. Although training processes all three positions together, the causal mask prevents the first position from reading the later `h` or `i`. The answer is present elsewhere in the batch, but cannot enter the calculation that predicts it.

[[diagram:prefix]]

The arrows in this board describe information used for a prediction and supervision used to measure it. A label is not fed back into its own prediction during the same forward call. At inference the next chosen token does become part of the next call's prefix.

[[practice:shift]]

### Why next-token prediction defines a language model

Conditional probability means “probability given some information.” Write $p(b\mid a)$ for the probability of b after observing a. For a token sequence, the probability chain rule gives

$$p(t_1,\ldots,t_n\mid BOS)=\prod_{i=1}^{n}p(t_i\mid BOS,t_1,\ldots,t_{i-1}).$$

The chain rule does not assume tokens are independent. Each factor has its own context. Our finite model uses at most its declared context window, so it approximates a distribution with bounded memory. Include EOS if you are scoring the probability of ending the document as well.

Suppose the three correct next-token probabilities are 0.5, 0.25 and 0.125. Their product is 0.015625. Training with that product directly is awkward: long sequences give extremely small products. The natural logarithm turns multiplication into addition: $\log(ab)=\log a+\log b$. Negating the log turns a high correct-token probability into a small loss.

$$L=-\frac{\log(0.5)+\log(0.25)+\log(0.125)}{3}
=\frac{0.693147+1.386294+2.079442}{3}=1.386294.$$

A natural-log loss is measured in **nats per predicted token**. Exponentiating gives perplexity: $\exp(L)=4$. This equals four equally likely choices per prediction in an equivalent constant-uncertainty case. It does not mean the vocabulary contains four tokens, nor that the model literally considered exactly four choices at every step.

[[practice:perplexity]]

Run [the small arithmetic program](examples/math_trace.py) with `python3 math_trace.py` after saving it in a scratch folder. It uses no Torch. Expect the shifted arrays, perplexity 4, weighted loss 2 and colder-sampling probability 0.9. The later lessons explain its GELU and temperature calculations. You can also save [the C++17 trace](examples/math_trace.cpp), compile with `g++ -std=c++17 math_trace.cpp -o math_trace`, and run `./math_trace` to check explicit memory indexing.

### What carries over from P00, and what changes

You retain the tokenizer artifact idea, causal attention, shape/gradient checks and complete-state recovery discipline. You replace the conditional task with text continuation. You remove encoder/cross-attention, use learned position embeddings, put LayerNorm before each sublayer, add a final LayerNorm, and use GELU in the feed-forward network. Those are explicit design choices, not consequences that follow automatically from deleting an encoder.

This required P01 course is an educational GPT-style model. Historical GPT-1 used a different post-norm/classification design; those optional architecture details are outside this required course. The next lessons teach every change beside the file that implements it.

Before starting the implementation, use P00's exit record to confirm its required learner gates. This new tutorial does not mark them complete. If you are previewing P01 early, keep reading this first calculation; return to the first unfinished P00 gate before treating P01 as your active model project.

### Your first session ends here

Without looking back, write x and y for `ok`, explain why position zero cannot read the answer, and say why lowering mean negative log probability lowers perplexity. If those connections are clear, the next lesson gives this calculation a home on disk.

[Continue to Lesson 2](lessons/02-workspace.md) · [Implementation checklist](project/implementation.md) · [Readiness](project/readiness.md) · [Sources](research.md)
