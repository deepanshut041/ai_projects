## 1. The whole picture, then the first attention calculation

We want a model that receives an English sentence and produces its French translation. For the sentence **“We read books daily”**, a possible target is **“Nous lisons des livres quotidiennement.”** Keep that whole job in view while we open up the model. We will first see how the pieces work together, then examine one operation inside them, and finally return to the complete design.

### The whole translation process

The model needs to do two connected jobs. First, it must turn the English input into information useful for producing the translation. That information must preserve more than a list of isolated words: the relationships between positions matter too.

Second, it must use that source information to build the French output. It generates a piece of text, called a **token**, then uses the output so far when choosing the next token. Tokens can be words or parts of words; use whole words in this first picture so the process is easy to follow.

The English source is available before translation begins. The French output grows as generation proceeds. After producing “Nous”, the model must decide what comes next using both the English input and the French prefix it has already produced. After producing “lisons”, that token joins the prefix for the following decision.

![The whole translation process: read the source, generate the next token and extend the output](images/20-translation-task.svg)

We call the part that builds source representations the **encoder**. We call the part that uses them and the target prefix to build the output the **decoder**. These names label the two jobs in the picture. We will open their internal computations only as the explanation needs them.

### Where attention fits in that picture

Both jobs need a way to select and combine information. While building a source representation, one position can benefit from information at other source positions. While producing a target token, the model can benefit from its available target prefix and the encoded source. **Attention is the operation we will study for making those connections.**

At a broad level it does three things: compare a receiving position with available positions, turn those comparisons into contribution weights, then combine the contributed information into a new vector. Later network operations use that vector. The same basic calculation can serve different places in the translator by changing where its inputs come from and which connections are allowed.

Training gives the model examples of source sentences and reference translations. It compares next-token predictions with the reference and adjusts the shared parameters controlling its calculations. That is how the comparisons and contributions can become useful for translation. We will follow that learning signal after understanding the forward calculation.

### How we will open the model, one level at a time

| Pass through the explanation | What we build on |
| --- | --- |
| This chapter: one information-sharing step | Start with the whole task, then inspect input rows and a weighted contribution |
| Chapters 2–3: choose information for each position | Calculate the weights from the input, then repeat the understood calculation for every receiver |
| Chapters 4–5: make the choices flexible | Examine learnable transformations and several separately weighted mixtures |
| Chapters 6–8: place the operation in the translator | Restrict unavailable information, then connect the target side to the source side |
| Chapters 9–10: connect learning and implementation | Follow the error back to parameters, then trace the actual reusable class |
| Chapter 11: return to the whole architecture | Reassemble the roles after their internal operations are understood |

Each pass answers a question left by the previous one. For now, we will narrow our attention to source representations and build one information-sharing calculation. We will return to target generation when the same operation is understood.

The local project already implements attention; its tokenizer, full encoder/decoder and translation trainer remain future work. The overview locates the component we are studying within the intended system. The final setup page is an optional run reference.

### First zoom: how text becomes rows of numbers

A tokenizer splits text into tokens and assigns each token an integer ID. An embedding table uses each ID to select a learned vector. The integer is an address in that table, not a numerical measure of the word. Real tokenizers can split a word into several tokens; for this diagram only, use one token per displayed word.

The first encoder layer receives one vector per source token, with positional information added. Stack these rows into a matrix called **X**. Here there are four displayed tokens. With d features per token, X has shape `(4,d)`. At later encoder layers, the rows are the preceding layer's outputs instead of fresh embeddings.

We have moved from the whole translation process to the input of one source-side layer. Its rows are numerical representations, and the operation we study will return numerical representations. Keep that level separate from the decoder's eventual next-token prediction while we inspect what happens between these rows.

### Why processing each row separately is insufficient

Suppose we apply the same neural network f independently to every source row:

$$h_0=f(x_0),\qquad h_1=f(x_1),\qquad h_2=f(x_2),\qquad h_3=f(x_3).$$

The row for “read” is transformed, but the calculation of that row has never read the rows for “We”, “books” or “daily”. It can learn complicated functions of its own input features and still have no route for receiving information from those other positions.

There is a direct way to check this claim: keep x_1 fixed and change x_0. The expression `h_1 = f(x_1)` cannot change. Stacking more row-independent layers does not create the missing connection. We need a computation in which an output at position i can depend on input rows at other positions j.

This distinction explains **context** in this course. A contextual representation has been computed using information from other permitted positions. Calling a vector an embedding or making it wider does not, by itself, create that dependency.

### How could we add communication?

One option is to average all source vectors and give that summary to every position. This does transmit information across rows, but every position receives the same mixture. A position cannot choose a different source of information because its own state has no influence on the weights.

A recurrent network offers another route: it passes a state through positions in sequence. Attention was also used in recurrent encoder–decoder translation models to let a decoder read different source states at different output steps, instead of relying only on one fixed source summary. The Transformer goes further: its self-attention layers let positions exchange information directly, without a recurrence through token positions. These are different developments; the Transformer did not invent the general attention idea. [Earlier attention motivation](https://arxiv.org/abs/1409.0473), [Transformer design](https://arxiv.org/html/1706.03762v7).

For our encoder, the design we will examine is this: **keep the sequence of rows, and let each position receive its own weighted combination of information from the available rows.** The weights will depend on the input. Several positions can calculate their combinations in parallel because they read the layer's input rows, rather than waiting for one another's newly computed outputs.

![Independent row processing versus information gathered from other rows](images/21-communication.svg)

### First understand what a contribution means

Before choosing the weights, inspect what they do. We will use one small numerical fixture throughout the course. It stands in for four source-position representations inside a layer:

| Source position | Input row X | Information available to contribute, V |
| --- | --- | --- |
| 0, “We” | `[1,0]` | `[2,0]` |
| 1, “read” | `[0,1]` | `[0,2]` |
| 2, “books” | `[1,1]` | `[2,2]` |
| 3, “daily” | `[0,0]` | `[0,0]` |

These four rows show all four pairs made from 0 and 1, including `[0,0]`. We chose them to make the arithmetic easy to inspect; attention does not require every pair to appear, and real embeddings can contain arbitrary real numbers. The zero row is a valid input in this fixture, not a padding marker.

These are chosen numbers for checking arithmetic. The labels track which source row owns which vector; the coordinates are not trained representations of those words. We use `V = 2X` here, and will make that transformation and the selection rule explicit in the next chapters. A contributed vector is called a **value**.

Suppose one receiving position is assigned weights `[0.4,0.2,0.3,0.1]`. Its gathered vector is:

$$c=0.4[2,0]+0.2[0,2]+0.3[2,2]+0.1[0,0]=[1.4,1.0].$$

The contribution from position 0 is `[0.8,0]`; position 1 contributes `[0,0.4]`; position 2 contributes `[0.6,0.6]`; position 3 contributes `[0,0]`. Its weight is 0.1, but multiplying a zero value still gives zeros. Add all four contributions coordinate by coordinate. One scalar weight controls the **whole value row**. The result has two features because the contributed values have two features.

This vector is often called the **context vector** for that receiving position. The word “context” now names a calculation we can point to: information from several source rows has entered one output row. Nothing in `[1.4,1.0]` is a token ID or a French word.

### Run exactly that calculation

The code uses PyTorch as a numerical array library. `weights[:, None]` makes a column so each scalar multiplies its matching value row. `sum(dim=0)` adds the rows. The final matrix product is the same weighted sum in compact notation.

```python
import torch

values = torch.tensor([[2., 0.], [0., 2.], [2., 2.], [0., 0.]], dtype=torch.float64)
weights = torch.tensor([0.4, 0.2, 0.3, 0.1], dtype=torch.float64)

contributions = weights[:, None] * values
context = contributions.sum(dim=0)

print(contributions)
print(context)
print(weights @ values)
```

An ordinary average of these values gives `[1,1]`. Selecting position 0 alone gives `[2,0]`. Our supplied weights produce `[1.4,1]`. Those are three different ways to gather the same available information.

The unresolved design question is now precise: **how should a receiving position calculate its weights from the input?** Hand-written weights cannot adapt to a new sentence. The next chapter derives the query/key comparison that makes the mixture depend on the current representations.

**Runnable copy:** [weighted sum](examples/walkthrough/01_weighted.py). [Setup and run commands](lessons/02-project-setup.md) are a separate reference.

[[practice:weighted-check]]
