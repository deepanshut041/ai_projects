"""Rebuild the course from its whole-course review; retain exact runnable blocks."""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path('/mnt/projects/ai')
COURSE = ROOT / 'ai_projects/docs/tutorials/projects/attention-is-all-you-need/attention-lab/v1'
BEFORE = ROOT / '.ai/attention-course-review/2026-10-26/before/course'
RECORD = Path(__file__).parent
PAGES = {}

PAGES['01-weighted-information.md'] = r'''## 1. The whole picture, then the first attention calculation

We want a model that receives an English sentence and produces its French translation. For the sentence **“We read books”**, a possible target is **“Nous lisons des livres.”** Keep that whole job in view while we open up the model. We will first see how the pieces work together, then examine one operation inside them, and finally return to the complete design.

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

The first encoder layer receives one vector per source token, with positional information added. Stack these rows into a matrix called **X**. If there are three tokens and each vector has d features, X has shape `(3,d)`. At later encoder layers, the rows are the preceding layer's outputs instead of fresh embeddings.

We have moved from the whole translation process to the input of one source-side layer. Its rows are numerical representations, and the operation we study will return numerical representations. Keep that level separate from the decoder's eventual next-token prediction while we inspect what happens between these rows.

### Why processing each row separately is insufficient

Suppose we apply the same neural network f independently to every source row:

$$h_0=f(x_0),\qquad h_1=f(x_1),\qquad h_2=f(x_2).$$

The row for “read” is transformed, but the calculation of that row has never read the rows for “We” or “books”. It can learn complicated functions of its own input features and still have no route for receiving information from those other positions.

There is a direct way to check this claim: keep x_1 fixed and change x_0. The expression `h_1 = f(x_1)` cannot change. Stacking more row-independent layers does not create the missing connection. We need a computation in which an output at position i can depend on input rows at other positions j.

This distinction explains **context** in this course. A contextual representation has been computed using information from other permitted positions. Calling a vector an embedding or making it wider does not, by itself, create that dependency.

### How could we add communication?

One option is to average all source vectors and give that summary to every position. This does transmit information across rows, but every position receives the same mixture. A position cannot choose a different source of information because its own state has no influence on the weights.

A recurrent network offers another route: it passes a state through positions in sequence. Attention was also used in recurrent encoder–decoder translation models to let a decoder read different source states at different output steps, instead of relying only on one fixed source summary. The Transformer goes further: its self-attention layers let positions exchange information directly, without a recurrence through token positions. These are different developments; the Transformer did not invent the general attention idea. [Earlier attention motivation](https://arxiv.org/abs/1409.0473), [Transformer design](https://arxiv.org/html/1706.03762v7).

For our encoder, the design we will examine is this: **keep the sequence of rows, and let each position receive its own weighted combination of information from the available rows.** The weights will depend on the input. Several positions can calculate their combinations in parallel because they read the layer's input rows, rather than waiting for one another's newly computed outputs.

![Independent row processing versus information gathered from other rows](images/21-communication.svg)

### First understand what a contribution means

Before choosing the weights, inspect what they do. We will use one small numerical fixture throughout the course. It stands in for three source-position representations inside a layer:

| Source position | Input row X | Information available to contribute, V |
| --- | --- | --- |
| 0, “We” | `[1,0]` | `[2,0]` |
| 1, “read” | `[0,1]` | `[0,2]` |
| 2, “books” | `[1,1]` | `[2,2]` |

These are chosen numbers for checking arithmetic. The labels track which source row owns which vector; the coordinates are not trained representations of those words. We use `V = 2X` here, and will make that transformation and the selection rule explicit in the next chapters. A contributed vector is called a **value**.

Suppose one receiving position is assigned weights `[0.5,0.25,0.25]`. Its gathered vector is:

$$c=0.5[2,0]+0.25[0,2]+0.25[2,2]=[1.5,1.0].$$

The contribution from position 0 is `[1,0]`; position 1 contributes `[0,0.5]`; position 2 contributes `[0.5,0.5]`. Add those three contributions coordinate by coordinate. One scalar weight controls the **whole value row**. The result has two features because the contributed values have two features.

This vector is often called the **context vector** for that receiving position. The word “context” now names a calculation we can point to: information from several source rows has entered one output row. Nothing in `[1.5,1.0]` is a token ID or a French word.

### Run exactly that calculation

The code uses PyTorch as a numerical array library. `weights[:, None]` makes a column so each scalar multiplies its matching value row. `sum(dim=0)` adds the rows. The final matrix product is the same weighted sum in compact notation.

@@PROGRAM@@

An ordinary average of these values gives `[4/3,4/3]`. Selecting position 0 alone gives `[2,0]`. Our supplied weights produce `[1.5,1]`. Those are three different ways to gather the same available information.

The unresolved design question is now precise: **how should a receiving position calculate its weights from the input?** Hand-written weights cannot adapt to a new sentence. The next chapter derives the query/key comparison that makes the mixture depend on the current representations.

**Runnable copy:** [weighted sum](examples/walkthrough/01_weighted.py). [Setup and run commands](lessons/02-project-setup.md) are a separate reference.

[[practice:weighted-check]]
'''

PAGES['04-scores-and-softmax.md'] = r'''## 2. Derive one attention output from the input rows

Chapter 1 showed how a position can gather information once contribution weights are supplied. We now open the part that was missing: how to calculate those weights from the current input. Stay with one receiving position first; chapter 3 will repeat the same calculation for the whole sequence.

Our source-side layer receives X, the input rows established in chapter 1. We want an output at position i that gathers useful information from other source positions. To calculate it, we need two things: a rule for assigning a weight to each source position, and the information that position will contribute after it receives a weight.

These are separate jobs. The features useful for deciding **whether to read a row** need not be the features useful for **what to pass onward**. We therefore give each input row three transformed versions.

### Where queries, keys and values come from

For input row x_i, calculate `q_i = x_i W_Q`, `k_i = x_i W_K` and `v_i = x_i W_V`. The W matrices are model parameters, shared across all positions. They will be adjusted during training. For one receiving position i, use:

| Vector | Comes from | Job in this output |
| --- | --- | --- |
| Query q_i | The receiving row x_i through W_Q | Supplies one side of every comparison for receiver i |
| Key k_j | Available row x_j through W_K | Supplies the other side of the comparison with position j |
| Value v_j | That same available row x_j through W_V | Supplies the features multiplied by position j's eventual weight |

A score against key j controls the contribution from value j. Keep that pairing: a high score against key 2 does not select value 0. “Query” and “key” name their roles in this comparison; no explicit question sentence or database is needed.

For the retained fixture, set `W_Q = I`, `W_K = I`, `W_V = 2I`, where I is the identity matrix. This gives exactly the X and V from chapter 1. We are choosing parameter settings so we can inspect one computation, rather than claiming that these settings were learned.

![One receiver compared with every key, then gathering the paired values](images/22-one-query.svg)

### Trace position 0 all the way through

Its query is `q_0 = [1,0]`. The keys are `[1,0]`, `[0,1]`, `[1,1]`. A dot product combines matching feature products into one scalar:

$$q_0\cdot k_0=1\times1+0\times0=1.$$

The three comparisons are `[1,0,1]`. Position 1 has a lower score; positions 0 and 2 tie. A score is a learned compatibility calculation. It is not a dictionary-similarity label or a probability yet.

Scaled dot-product attention divides each score by the square root of the query/key feature width. Our width is two, so the scaled row is `[0.707107,0,0.707107]`. To see why this factor exists, suppose the coordinates of q and k are independent, zero-mean and unit-variance at initialization. Summing d products gives dot-product variance d. Dividing by `sqrt(d)` keeps that variance at one under those assumptions. Otherwise, wider comparisons can start with very large score differences and nearly saturated softmax weights. This is an initialization rationale, not a guarantee about every trained vector.

### Convert relative scores into contribution weights

We need a separate set of nonnegative weights for this receiver, with total weight one. Softmax takes each exponentiated score and divides by the total for this row:

$$a_{0j}=\frac{\exp(s_{0j})}{\exp(s_{00})+\exp(s_{01})+\exp(s_{02})}.$$

The exponentials are about `[2.028115,1,2.028115]`. Their total is `5.056230`. The resulting weights are `[0.401112,0.197776,0.401112]`.

Now complete the weighted sum from chapter 1, using these calculated weights:

| Available position | Weight | Weighted value contribution |
| --- | --- | --- |
| 0 | 0.401112 | approximately `[0.802224,0]` |
| 1 | 0.197776 | approximately `[0,0.395552]` |
| 2 | 0.401112 | approximately `[0.802224,0.802224]` |

Adding the unrounded contributions gives `c_0 = [1.604448,1.197776]`. This output has acquired information from other positions, but its relative contributions depend on the receiving row through q_0. That is the input-dependent communication we wanted.

### Implement the same chain

The program writes this one-query calculation directly. `query @ keys.T` performs all three comparisons. Subtracting the maximum score before exponentiating changes neither the final weights nor their ratios: the common factor cancels in the numerator and denominator. It avoids needlessly large exponentials.

@@PROGRAM@@

### What changes when the receiving state changes?

Hold K and V fixed and use query `[0,1]`. Scores become `[0,1,1]`, and the gathered result becomes `[1.197776,1.604448]`. The memory is identical; the receiver's comparison changed. Query `[0,0]` gives equal scores and recovers ordinary averaging.

This also explains why Q/K/V are useful roles rather than three interchangeable names for an embedding. Changing Q can redistribute weights. Changing V with Q and K fixed changes the content without redistributing weights. Chapter 4 examines the parameter maps that control those choices.

We used only one receiving row here. The encoder must retain an output for every source position, so next we arrange the same calculation into matrices. The formula and scaling follow [the original paper, section 3.2.1](https://arxiv.org/html/1706.03762v7).

**Runnable copy:** [one query through scores, weights and contributions](examples/walkthrough/02_softmax.py).

[[practice:softmax-check]]
'''

PAGES['03-tensors-and-products.md'] = r'''## 3. Keep one output per position and expose the matrix axes

One query gave position 0 a gathered vector. The encoder cannot stop with that one result: the next layer needs a sequence of representations, including an updated row for positions 1 and 2. Repeat the same procedure with each input row supplying its own query.

All queries read the **same input** keys and values. Position 1 does not read position 0's newly computed output. This matters because it lets us calculate every row at once.

### Each score cell has an owner

Stack the queries into Q, keys into K and values into V. In our identity/doubling fixture, `Q=X`, `K=X`, `V=2X`. The product `Q @ K.T` is:

```text
                         source key
                        0    1    2
query 0, receives       [1    0    1]
query 1, receives       [0    1    1]
query 2, receives       [1    1    2]
```

Cell `(i,j)` compares receiver i with source j. A row answers “how does this receiver score all available positions?” A column answers “how do different receivers score this one available position?” We need the first question to obtain a mixture for each receiver, so softmax normalizes **across columns within each row**.

![The score matrix, row-normalized weights and output rows with their dimensions](images/23-matrix-axes.svg)

After scaling and softmax, the weight rows are approximately:

```text
A = [[0.401112, 0.197776, 0.401112],
     [0.197776, 0.401112, 0.401112],
     [0.248255, 0.248255, 0.503490]]
```

The product `C = A @ V` sums over the available-position axis. It preserves the receiver axis and the value-feature axis:

$$C_{im}=\sum_j A_{ij}V_{jm}.$$

For example, `C[0,1]` is the **second feature received by query 0**. Its calculation is `A[0,0]×0 + A[0,1]×2 + A[0,2]×2`. No position disappeared merely because matrix multiplication was used.

### Put the repeated computation in one function

This function accepts queries and keys with the same comparison width, and paired keys/values with the same source-position count. All keys are available for now. `keepdim=True` retains a size-one column for each row's maximum and denominator, so each receiver uses its own normalization.

@@PROGRAM@@

The output rows are `[1.604448,1.197776]`, `[1.197776,1.604448]` and `[1.503490,1.503490]`. They have different mixtures even though they read the same source rows. The first row is exactly chapter 2's calculation, now embedded in the full matrix.

### Batch sequences without mixing them

`unsqueeze(0)` adds a batch axis: `(3,2)` becomes `(1,3,2)`. The assertion checks that this wrapper preserves the result. For a batch of B independent examples, use:

| Quantity | Shape | What the axes mean |
| --- | --- | --- |
| Q | `(B,T,d_k)` | example, receiving position, comparison feature |
| K | `(B,S,d_k)` | example, available position, comparison feature |
| V | `(B,S,d_v)` | example, available position, contributed feature |
| Scores and A | `(B,T,S)` | example, receiver, available position |
| C | `(B,T,d_v)` | example, receiver, received feature |

The batch axis is carried through independently. It must never be flattened into S, which would allow one sentence to read another sentence's tokens. Query count T and source count S can differ; cross-attention will use that freedom.

### Where does word order enter?

There is a missing capability if we feed only token embeddings into this operation: unrestricted self-attention has no inherent notion of first, second or third. Reorder X's rows and its output rows reorder in the same way. Pairwise scores depend on vector content, not an invisible row-number feature.

More precisely, for a permutation matrix P and no masks or positional signals, `Attention(PQ,PK,PV) = P Attention(Q,K,V)`. The matching keys and values move together. The operation treats a sequence as a collection of rows with corresponding outputs.

Translation needs order. The original Transformer adds positional encodings to token embeddings before the encoder/decoder stacks. Then the input rows carry position information, so reordering tokens while assigning them their new positions changes more than a simple row permutation. The [paper's section 3.5](https://arxiv.org/html/1706.03762v7) specifies that component. Our numerical fixture isolates attention; it does not calculate positional encodings.

The matrix calculation exposes another tradeoff: a length-n sequence has n² pair comparisons per head. Scores and weights can be computed in parallel, but parallelism does not remove that quadratic work or, in the explicit implementation, storage. This is why “can connect distant positions directly” should not be read as “free for arbitrarily long inputs”.

**Runnable copy:** [all positions and batch check](examples/walkthrough/03_shapes.py).

[[practice:shapes-check]]
'''

PAGES['05-learned-projections.md'] = r'''## 4. What the learned projections can change

We can now trace an input matrix to one output row per position. Our parameter choices were simple: Q and K copy X, while V doubles it. A useful model needs to adapt both its comparison rule and the information carried by each contribution. This is the responsibility of the projection parameters.

### Separate model parameters from this input's activations

$$Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.$$

W_Q, W_K and W_V persist between examples. Q, K, V, scores, attention weights A and context C are calculated anew for each input. In particular, the entries of A are **not a table of persistent parameters indexed by sentence positions**. A is produced by the current input and learned projections.

Each W is shared across positions. For a model width d and a single head with comparison width d_k, W_Q and W_K have shape `(d,d_k)`; W_V has shape `(d,d_v)`. The matrices do not acquire extra parameter rows when the sentence becomes longer. The sequence length controls the number of activation rows and pair comparisons.

### Why not use the same transformed vector for every job?

Consider one score before scaling:

$$q_i k_j^T = x_i W_Q W_K^T x_j^T.$$

The receiving row and available row enter through different matrices. This permits an asymmetric relation: position i reading j need not receive the same score as j reading i. If Q=K, the raw self-attention score matrix is symmetric; its row-normalized weights still need not be symmetric because different rows can have different denominators. Separate query/key maps remove the raw-score symmetry constraint as well.

The value projection has a different purpose. Once the comparison assigns weight A_ij, V decides which features travel from source j to receiver i. Requiring the content features to equal the comparison features would unnecessarily tie those choices together. Separate maps provide capacity to learn them separately; they do not guarantee a particular linguistic specialization.

### A controlled change that isolates the value path

Use the identity/doubling settings again so the starting output stays recognizable. Then halve only W_V. Q, K, scores and A are untouched. V halves, so every C row halves. That is a direct consequence of `C = A @ V`, rather than a new selection decision.

@@PROGRAM@@

The original first context remains `[1.604448,1.197776]`. After the isolated value change it is approximately `[0.802224,0.598888]`, with exactly the same attention weights. Changing X instead would generally affect all three roles at once, so it would be a poor way to isolate the value projection.

Changing W_Q can change several receivers because the same matrix transforms all their rows. In chapter 9, the first input row `[1,0]` produces a gradient on W_Q's first row. That update can also affect the query for input row `[1,1]`; it is not a private parameter belonging only to token position 0.

### Connect the notation to the implementation

Our handwritten notation multiplies a row on the right, `x @ W`. PyTorch's `nn.Linear` stores its matrix in the transposed convention and evaluates `x @ weight.T + bias`. To copy a general handwritten W into the layer, copy `W.T`. Identity matrices hide that distinction, so it is worth making the orientation explicit now.

The project's `q_proj`, `k_proj`, `v_proj` and `out_proj` are affine layers with biases. Their parameters become useful only when a loss and optimizer adjust them. Calling a forward pass with identity maps demonstrates the computation; it does not train those maps. We will trace the learning path after locating encoder and decoder attention.

These separate maps also make multiple heads possible. Instead of forcing every output feature to use one shared distribution over source positions, we can compute several distributions in projected feature groups. That is the next design question.

**Runnable copy:** [projections and isolated value change](examples/walkthrough/04_projections.py).

[[practice:projections-check]]
'''

PAGES['06-multiple-heads.md'] = r'''## 5. Why use several attention heads?

One head produces one distribution over source positions for each receiver. In our example, receiver 0 uses weights `[0.401112,0.197776,0.401112]`. The first output feature and the second output feature both use that same distribution. W_V can change the content carried by a source row, but within this head the row still has one scalar contribution weight.

That is the restriction multiple heads relax. A layer can give different projected feature groups independently calculated distributions over positions, then combine the gathered features. It does not have to compress every relationship into one mixture.

### What a head owns

Each head has query, key and value features, its own scores and its own softmax distribution for each receiver. Every head can still read every permitted position. Heads divide projected **features**, not the sentence into segments.

In the implementation, a single full-width projection produces all head features together, then a reshape labels which features belong to which head. This is equivalent to collecting separate per-head projection matrices side by side. Since projection happens before splitting, each head's features can depend on the whole input row.

### Work the two heads for the same receiving position

Keep X and the identity/doubling projections, but use two heads of width one. For receiver 0:

| Calculation | Head 0 | Head 1 |
| --- | --- | --- |
| Query feature | `[1]` | `[0]` |
| Keys at positions 0, 1, 2 | `[1], [0], [1]` | `[0], [1], [1]` |
| Values at positions 0, 1, 2 | `[2], [0], [2]` | `[0], [2], [2]` |
| Scores, scaled by sqrt(1) | `[1,0,1]` | `[0,0,0]` |
| Weights | `[0.422319,0.155362,0.422319]` | `[1/3,1/3,1/3]` |
| Gathered feature | `1.689275` | `1.333333` |

The second head averages because this particular receiver's second query feature is zero. It is not assigned a special language task. The first head prefers positions 0 and 2. The two gathered features therefore come from different mixtures of the same source positions.

Concatenate them to get `[1.689275,1.333333]`, then apply W_O, the output projection. W_O is identity in this check so the two gathered features stay visible. In a trained model it can mix them into the next layer's feature space. It does not recalculate their attention weights.

![One distribution in one head versus separately normalized distributions in two heads](images/24-head-layout.svg)

### Follow the axes through the implementation

Start with `(B,T,d_model)`. Reshape to `(B,T,H,d_head)` and transpose to `(B,H,T,d_head)`. Now each `(batch,head)` pair holds an ordinary position-by-feature attention problem. After gathering, transpose back to `(B,T,H,d_head)` **before** joining head features into each position's row.

@@PROGRAM@@

The output rows are approximately `[1.689275,1.333333]`, `[1.333333,1.689275]`, `[1.689275,1.689275]`. They differ from the one-head results because the comparison features, per-head scaling and normalization groups changed. The reshape alone would not justify an improvement claim.

If you flatten while heads still precede positions, elements can end up in the wrong token row even though the final shape looks plausible. This is why the inverse transpose is part of correctness rather than cosmetic layout code.

At fixed model width, increasing H reduces each head's width. More independently weighted mixtures come with narrower per-head feature spaces. Neither a larger head count nor a diagram assigning “grammar” to one head and “meaning” to another establishes a better model. The paper motivates multiple representation subspaces; useful specializations and task quality require trained-model evidence. [Paper, section 3.2.2](https://arxiv.org/html/1706.03762v7).

**Runnable copy:** [two heads with visible score and value paths](examples/walkthrough/05_heads.py).

[[practice:heads-check]]
'''

PAGES['07-padding-and-empty-rows.md'] = r'''## 6. Padding: exclude storage rows from the information flow

We have built the gathering calculation for every position and allowed several independently weighted mixtures. So far every stored source row has been treated as available information. We now return to the translator around that calculation: batching introduces stored rows that do not belong to a sentence, so the information flow needs a boundary.

The encoder will process batches of sentences. Our source has three displayed tokens; another source in the batch might have only two. A rectangular `(batch,positions,features)` tensor still needs the same stored position count for both examples, so the shorter one receives a padding row.

```text
Example A: real token | real token | real token
Example B: real token | real token | PAD
Validity B:    True   |    True    | False
```

Padding occupies memory but is not part of the source sentence. If attention treats it as another source position, it can alter the representations of real tokens. We must prevent that dependency.

### Even a zero value can contaminate an output

The problem is not solved by making the padded value zero. Its key still contributes a score and an exponential to the softmax denominator. Some weight is spent on the padded position, reducing the weights assigned to real values. The output can change even though the padded value contributes no nonzero features directly.

The right boundary is therefore **before normalization**. A boolean allowed mask states which query/key pairs exist for the calculation. This project uses `True` for allowed. For example B, `[True,True,False]` excludes source key 2 for every receiver.

Replace forbidden scores with negative infinity. Their exponentials are zero, leaving only allowed positions in the denominator. For receiver 0, the scaled row changes from `[0.707107,0,0.707107]` to `[0.707107,0,-inf]`. The weights become `[0.669762,0.330238,0]` and context becomes `[1.339523,0.660477]`.

Masking after softmax without renormalizing would keep the old weights on valid keys; masking the final context would be later still. Neither implements this normalized distribution over the allowed positions.

### Define what happens when nothing is allowed

Softmax of an all-negative-infinity row is undefined: subtracting its maximum encounters `-inf - (-inf)`. The project's explicit policy is zero weights and zero gathered context for a completely blocked receiver.

The small function below implements the same policy. It substitutes safe temporary scores for empty rows, performs a finite softmax, then replaces those weight rows with zeros. It also accepts a different allowed row per receiver, which we will need for decoder causality.

@@PROGRAM@@

For our fixture with source position 2 masked, contexts are `[[1.339523,0.660477],[0.660477,1.339523],[1,1]]`. The final empty-memory check returns `[[0,0]]`.

### Prove the excluded row cannot affect real receivers

The program changes the padded row to `[1000,-1000]`, then recomputes attention. The first two outputs must stay unchanged. This checks the intended absence of influence, not merely the presence of a triangular picture or a zero in a printed tensor.

Only the first two receivers are compared because changing X[2] also changes query 2. Masking **key 2** removes that row as a source of information; it does not freeze or delete its query. The original third context `[1,1]` is therefore consistent: the padded query still reads the two real keys.

Keep three responsibilities separate. A key mask excludes source contributions. A target-loss mask later excludes padded prediction positions from the objective. An output projection can add its bias even when gathered context is zero. Confusing those boundaries leads to tests that look reassuring while checking the wrong behavior.

**Runnable copy:** [padding, perturbation and empty-row checks](examples/walkthrough/06_padding.py).

[[practice:padding-check]]
'''

PAGES['08-causal-information.md'] = r'''## 7. Decoder causality: match training to what generation can know

Return to our translation task. When generating “Nous lisons des livres”, the decoder has the source sentence, but it does not yet have the target tokens it is supposed to predict. After “Nous”, it must predict “lisons” using the prefix that exists so far.

During training, the complete reference translation is available. We would like to train all target positions together in a matrix, but we must prevent a position from using a future reference token as an input shortcut.

### Show the exact input and label alignment

BOS denotes a beginning token; EOS denotes an ending token. Continue to use whole displayed words only to show ownership; the eventual tokenizer decides actual token boundaries.

| Decoder position | Input token at this position | Desired next token | Allowed target-input prefix |
| --- | --- | --- | --- |
| 0 | BOS | Nous | BOS |
| 1 | Nous | lisons | BOS, Nous |
| 2 | lisons | des | BOS, Nous, lisons |
| 3 | des | livres | BOS, Nous, lisons, des |
| 4 | livres | EOS | BOS, Nous, lisons, des, livres |

At position 1, the label is “lisons”. Input position 2 contains exactly that token. Without a restriction, position 1 could read it through self-attention and make a prediction using information unavailable at generation time. This is the concrete leakage path the causal mask must remove.

The rule is `key_index <= query_index`. It allows the current input token because the prediction at that row is for the **next** token. The label shift and the mask work together; a mask does not repair unshifted labels.

![Training input, next-token labels and the allowed causal pairs](images/25-decoder-causality.svg)

### Apply the rule to the same three-row arithmetic fixture

To keep every number inspectable, use the first three-position mask pattern with our existing numerical X. These rows now stand in for decoder states; they are not literal embeddings of the tokens in the table.

```text
             key 0  key 1  key 2
query 0       True  False  False
query 1       True   True  False
query 2       True   True   True
```

Query 0 reads only value 0, so its gathered vector is `[2,0]`. Query 1 compares only keys 0 and 1 and receives `[0.660477,1.339523]`. Query 2 can use all keys, so it keeps `[1.503490,1.503490]`. The algorithm is the masked attention from chapter 6 with a different allowed matrix.

@@PROGRAM@@

### Check future independence separately from padding

The perturbation changes position 2 while keeping all three positions real. Outputs at positions 0 and 1 must remain unchanged. This tests causality itself. If padding already hid position 2, an unchanged result would not independently prove that the causal rule worked.

If position 2 is also padding, combine restrictions with AND: a key must be both real and no later than the query. Query 2 then reads positions 0 and 1 and returns `[1,1]`. The first two outputs keep their earlier causal results.

The source encoder usually can read the whole valid English sentence: those tokens were supplied before translation began. Target self-attention must obey the prefix boundary. Training can calculate many target rows together under that restriction; generation still constructs the next token sequentially. Parallel matrix computation during training does not imply that the complete translation is already known at inference time. [Paper, decoder stack and attention applications](https://arxiv.org/html/1706.03762v7).

**Runnable copy:** [causal pairs and future perturbation](examples/walkthrough/07_causal.py).

[[practice:causal-check]]
'''

PAGES['09-cross-attention.md'] = r'''## 8. Let the decoder read the encoded source

Causal self-attention lets a decoder position use its allowed French prefix. That alone provides no path from the English sentence to the next prediction. We need a second attention operation that reads the encoder's source representations.

For the row predicting “lisons”, the receiver is a decoder state built from its allowed prefix. The available memory consists of all valid encoded source rows. **Cross-attention** uses the decoder state to form Q and encoder memory to form K and V.

### Follow the two streams into one output

```text
encoded English source rows ── W_K ── K
                            └─ W_V ── V
causal French prefix states ── W_Q ── Q

Q K^T → source weights for each target receiver → weights V
```

The receiver's target position survives. The source-position axis is summed out. The gathered source features continue through decoder computation and eventually a vocabulary projection to support the next-token prediction. Cross-attention itself still returns features.

![Two target receivers reading three source memory rows](images/26-cross-attention.svg)

### Why source and target lengths need not match

Our displayed source has three words while its target has four. Translation is not a one-source-row-to-one-target-row mapping. In the small program, use three source states and two target states to expose the rectangular geometry. The target states are chosen vectors for two receiving rows, not outputs of an implemented French decoder.

With Q shaped `(2,2)` and K shaped `(3,2)`, scores and weights have shape `(2,3)`. Multiplying by V shaped `(3,2)` returns `(2,2)`: one gathered vector for each of the two target receivers.

@@PROGRAM@@

The first two familiar queries recover `[[1.604448,1.197776],[1.197776,1.604448]]`. Excluding source position 2 gives `[[1.339523,0.660477],[0.660477,1.339523]]`. The numerical operation did not change; the origin and ownership of its vectors did.

### Which masks belong here?

The source key mask has three entries because there are three memory rows. A two-entry target-validity row cannot serve as that mask. Target padding and target-loss masking are separate responsibilities.

Do not apply a target-time triangle over source indices. Source index 2 is not a future French token. The entire English source is available before generation. The decoder state supplying the query must already respect target causality; cross-attention may then read the full valid source. This is the encoder–decoder attention role described in [the paper's section 3.2.3](https://arxiv.org/html/1706.03762v7).

### Connect the roles to the actual call

The project calls the class as `layer(src, tgt=target_states)`. Here `src` means the memory that supplies K and V; `tgt` supplies Q. Without tgt, the same src supplies all three and the operation is self-attention. Output length follows tgt when it is supplied.

The reusable class requires matching batch size, feature width, dtype and device, while source and target sequence lengths can differ. It rejects `causal=True` with tgt supplied because that flag implements a self-attention triangle, not a cross-sequence time relation.

We have now located the three uses of attention: source-to-source, target-prefix-to-target-prefix, and target-to-source. Next we connect the translation objective to the parameters that control their comparisons and contributions.

**Runnable copy:** [rectangular cross-attention and source mask](examples/walkthrough/08_cross.py).

[[practice:cross-check]]
'''

PAGES['11-gradients-and-learning.md'] = r'''## 9. How the translation error reaches attention

Q/K/V projections were chosen by hand in our worked calculation. A trained translator must learn parameters that help predict the reference translation. To see where that learning signal comes from, follow the decoder beyond attention.

### Start with the actual task objective

For the target row whose prefix ends in “Nous”, the desired next token is “lisons”. Let h_t be that row's final decoder representation. A vocabulary projection calculates logits `z_t = h_t W_U + b_U`, one score for each possible target token. Softmax converts those scores into a next-token distribution p_t. Cross-entropy penalizes low probability on the reference token:

$$L_t=-\log p_t[\text{reference token}].$$

The derivative with respect to logits is `p_t - one_hot(reference token)`. That error flows backward through the vocabulary map, decoder computation, attention outputs, value contributions and query/key comparisons. W_Q, W_K and W_V can all receive gradients. No training file needs to say “this head should attend to source position 2”. Their selection behavior can be learned indirectly from the token-prediction objective.

![Task loss flowing back through decoder features to attention parameters](images/27-learning-path.svg)

This is the real-model learning path. The full translation model is not implemented locally yet. We will magnify one part of that path with a controlled objective on the gathered vector, so its derivative can be calculated completely instead of invoking a missing decoder.

### Isolate one retrieval error

Keep our first query, keys and values. For this derivative probe only, set a desired context `t = [2,0]`. The current context is `[1.604448,1.197776]`, so use half squared error:

$$L=\tfrac12\sum_m(c_m-t_m)^2\approx0.795564.$$

The error vector `r = c-t` is approximately `[-0.395552,1.197776]`. The first feature is too small and the second too large. Giving more relative weight to value `[2,0]` would help both errors. Increasing every score equally would do nothing, because the common shift cancels in softmax.

We are not assigning this desired context to the translation of “We”. It is a local objective chosen to expose the mechanism by which an error changes retrieval.

### Differentiate the computation, one dependency at a time

From the squared loss, `dL/dc = r`. Since `c = sum_j a_j v_j`, increasing weight a_j changes c in direction v_j, giving `dL/da_j = r dot v_j`.

Weights are coupled by the softmax denominator. Its derivative is `da_l/ds_j = a_l(delta_lj-a_j)`, where delta_lj is one when l=j and zero otherwise. Combining the terms gives:

$$\frac{\partial L}{\partial s_j}=a_j\,r\cdot(v_j-c).$$

The subtraction of c accounts for redistributing weight away from the current mixture when a source score increases. It is the reason we cannot treat each weight as an independent knob.

Finally, `s_j = q dot k_j / sqrt(2)`, so:

$$\nabla_q L=\sum_j\frac{\partial L}{\partial s_j}\frac{k_j}{\sqrt2}
\approx[-0.223132,0.451290].$$

This gradient describes how the chosen loss changes for small query changes. Gradient descent subtracts it: increase the first query coordinate and decrease the second. The direction follows from this fixture and objective, rather than being an intrinsic property of the word or query role.

### Update a shared parameter and inspect the consequence

The code makes W_Q trainable and holds W_K/W_V fixed, isolating the query path. `backward()` computes the gradient; it does not update parameters. The explicit subtraction inside `no_grad()` performs the update.

@@PROGRAM@@

Input row 0 is `[1,0]`, so it selects W_Q's first row. That row receives gradient `[-0.223132,0.451290]`; the second row receives zero in this one-query probe. A rate of 0.1 changes the query to about `[1.022313,-0.045129]`, the context to `[1.614461,1.179891]`, and the probe loss to `0.770391`.

In a full training step, gradients from many target positions and examples accumulate into shared parameter matrices before the optimizer update. A later example calculates new attention weights with the updated parameters. Attention weights are intermediate results, not parameters that the optimizer directly stores for each token pair.

The local loss decrease verifies this derivative path. Translation quality requires the actual token objective, complete model, data, repeated updates and held-out evaluation. We can now read the reusable class knowing both what its forward pass computes and how its parameters could receive a task signal.

**Runnable copy:** [one query-projection update](examples/walkthrough/09_gradient.py).

[[practice:gradient-check]]
'''

PAGES['10-reusable-attention.md'] = r'''## 10. Trace one call through the actual project implementation

The small programs exposed the attention calculation. The actual module adds a reusable interface: it validates input, owns projections, arranges heads, builds masks, selects an execution path and applies the output projection. Read it as that sequence of responsibilities.

The source is `src/core/attention.py`. We will run `layer(src, need_weights=True)` with `src` shaped `(1,3,2)`, one head, identity query/key/output maps, doubled value map and zero biases. Those settings must reproduce the three context rows already derived.

### First follow the data, before the complete source

Calling an `nn.Module` runs its `forward` through PyTorch's module machinery. In this class, that forward call takes the following path:

| Stage | Actual owner | Data and consequence in this call |
| --- | --- | --- |
| Validate | `_validate_inputs`, `_sequence` | Require a floating-point `(batch,nonempty sequence,2)` src. A supplied tgt must share batch, dtype and device; source validity must match src's first two axes. |
| Choose receiver | `Attention.forward` | No tgt was supplied, so `query_input = src`. With tgt supplied, only the query origin changes. |
| Project | `q_proj`, `k_proj`, `v_proj` | Produce Q=X, K=X, V=2X, each `(1,3,2)` here. The actual layers also own biases. |
| Arrange heads | `_heads` | Reshape then transpose each projection to `(1,1,3,2)`: batch, head, position, feature. Position 0's query stays `[1,0]`. |
| Build availability | `_context` → `make_attention_mask` | Neither padding nor causality was supplied, so the mask is None. Every query may read every key. |
| Calculate | `_context` → `scaled_dot_product_attention` | Requesting weights chooses the explicit path. Scores/weights have shape `(1,1,3,3)`; context has `(1,1,3,2)`. |
| Rejoin | `Attention.forward` | Transpose to put positions before heads, then flatten head features. Shape returns to `(1,3,2)`. |
| Output map | `out_proj` | Apply the identity output map and zero bias. Return output plus weights because they were requested. |

Follow just output `[0,0]`: the first query `[1,0]` scores the three keys, obtains `[0.401112,0.197776,0.401112]`, gathers `[1.604448,1.197776]`, and passes through the identity output map unchanged. The class is packaging the calculation we already understand.

### Follow a second call with a different information boundary

With source validity `[True,True,False]`, `make_attention_mask` expands the source key row to `(B,1,1,S)`. Broadcasting shares that rule across heads and receivers. A causal self-attention request additionally supplies a `(1,1,T,S)` lower triangle; AND combines the restrictions.

With tgt supplied, Q has tgt's position count while K/V retain src's position count. The output follows tgt. The class rejects causal cross-attention rather than interpreting a self-attention triangle as a relation between unrelated source and target indices.

These branches explain the API arguments. `src` is memory, `tgt` optionally supplies receivers, `attention_mask` describes valid source keys, and `causal` restricts self-attention's target-time information flow.

### Understand the explicit and library paths

The explicit helper computes scores, replaces disallowed entries, handles empty rows, applies softmax and gathers V. It returns weights from **before dropout**. During training, context uses dropped-and-rescaled weights, so the gathered result need not remain a convex average even though returned inspection weights sum to one on valid rows.

The optional PyTorch `F.scaled_dot_product_attention` path computes the same operation through a library implementation. `need_weights=True` forces the explicit branch so those intermediate weights can be inspected. Without a padding mask, the library can apply a causal rule internally; with padding, the code builds the combined mask. Evaluation explicitly passes dropout probability zero to the library. [PyTorch SDPA interface](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

To compare the paths, copy the **same parameters** into both layers and use the same inputs, mask, dtype and evaluation mode. Two independently randomized layers are different functions and are not a useful parity check. The program below performs the controlled comparison on CPU; it does not measure GPU speed.

### Complete exact definitions and executable call

Every project helper called by `Attention` is included below, in dependency order. `_dimensions` checks constructor head widths; the other helpers implement the call path above. This complete source is available here so the walkthrough does not depend on an unseen notebook cell or import from a missing file.

@@PROGRAM@@

The output is:

```text
[[1.604448, 1.197776],
 [1.197776, 1.604448],
 [1.503490, 1.503490]]
```

The final assertion compares the optimized call with this explicit result using the same state. The definitions are checked against the actual project source by syntax tree, not merely by their names.

For an all-blocked request, the helper returns zero gathered context. The later biased `out_proj` can still return its bias. Likewise a key mask does not delete padded query rows. These outcomes follow from where each responsibility sits in the call path.

**Runnable copy:** [complete actual class and controlled execution](examples/walkthrough/10_interface.py). The [captured source](source:attention-source) records the inspected implementation.

[[practice:interface-check]]
'''

PAGES['12-evidence-and-next-phase.md'] = r'''## 11. Put attention back into the Transformer

We began with an English sentence and a decoder that must generate its French translation using the source and an available target prefix. We can now name the three information-sharing operations in that system and trace their outputs.

| Location | Receiver Q | Available K/V | Information boundary |
| --- | --- | --- | --- |
| Encoder self-attention | Source representations entering the layer | The same source representations | All real source tokens are available |
| Decoder self-attention | Target representations entering the layer | The same target representations | Real target-prefix tokens through the current input position |
| Decoder cross-attention | Decoder states after its preceding sublayer | Final encoder memory | All real source tokens, regardless of target index |

The formula is shared, but vector origin and allowed connections determine its role. Describing a matrix only as “attention” without identifying those two facts leaves the important part unexplained.

### The surrounding network still has work to do

In the original 2017 Transformer, token embeddings and positional encodings enter stacked encoder/decoder layers. Encoder layers have self-attention and a position-wise feed-forward network. Decoder layers have causal self-attention, encoder–decoder attention and a position-wise feed-forward network. Sublayers use residual addition followed by LayerNorm in the original post-norm design. The final decoder representations pass through a vocabulary projection to predict the next token. [Paper, sections 3.1–3.5](https://arxiv.org/html/1706.03762v7).

Residual addition preserves a direct path for the receiving row alongside its computed update. The feed-forward network transforms features within each row; it does not replace attention's cross-position communication. Positional encodings supply order information that unrestricted attention alone lacks. LayerNorm controls per-row feature normalization. Those responsibilities make attention useful inside a model rather than turning the weighted-sum function into a complete translator.

The title “Attention Is All You Need” describes the paper's replacement of recurrent/convolutional sequence mixing. It does not mean that embeddings, positions, feed-forward layers, residuals, normalization, output predictions or training objectives disappear.

### What each construction actually resolved

An independent row function had no neighbor-to-output path. A weighted sum created such a path. Query/key projections made the weights depend on current receiving and source representations; the value projection chose what traveled. Matrix multiplication retained one result per receiver. Heads allowed different projected feature groups to use different distributions. Padding and causal masks removed invalid information paths. Cross-attention connected decoder receivers to source memory. A loss supplied gradients to the shared parameters.

These are the reasons for the operations, not just their execution order. To explain a layer, start from its input ownership and allowed information, then follow a score, a normalized contribution and an output row.

### The exact local implementation boundary

The existing project contains the attention module, its demo, CPU checks and a Colab execution notebook. The ten standalone programs in this course make the arithmetic and parameter behavior inspectable. They do not supply a tokenizer, training corpus, complete Transformer, translation trainer or measured translation quality.

The next implementation phase must establish actual source/target tokenization and batches: dataset/split identity, special tokens, input/label shift, source-key validity and target-loss exclusion. After that come the full model, training, decoding, evaluation and persistent recovery. Actual hosted execution remains a separate check.

The conceptual attention route ends here. The following setup page is a run/reference page. The [project return page](project/implementation.md), [learning points](project/learning-points/01-attention.md), [comparison review](project/comparisons/01-attention.md), [optional review questions](practice/explain-attention.md), [readiness](project/readiness.md) and [source/check record](project/reference-runs.md) support a return visit. The [whole-course teaching review](project/teaching-review.md) explains why this route was rebuilt.

The previous Attention videos were removed from this tutorial and its project notes at the learner's request on 26 October 2026. This rewritten explanation is available for review; technical verification does not establish its teaching acceptance or the learner's understanding.

[[practice:handoff-check]]
'''

PAGES['02-project-setup.md'] = r'''## Reference — run and inspect the existing attention project

Use this page when you want to execute the calculation from [the first lesson](lessons/01-weighted-information.md). The conceptual route is readable without environment setup.

### Run the implemented component and all existing checks

```bash
cd /mnt/projects/ai/ai_projects/projects/transformer-labs/attention-is-all-you-need
uv sync --frozen --extra notebook
uv run --frozen python src/attention.py
uv run --frozen --extra notebook pytest -q
```

The project is an existing ordinary folder tracked by `ai_projects`. `uv sync --frozen` uses its lockfile rather than resolving a new dependency set. The notebook extra supplies the visualization dependencies so the full set includes all 25 attention and 10 visualization-maths cases. A skipped visualization module does not check those ten cases.

The locked environment uses Python 3.12.13 and CPU PyTorch 2.14.1. A GPU being installed does not change the selected runtime. This command runs a component demo, not a translator.

### Run one complete teaching program

The ten displayed Python programs include their own imports, inputs and called helper definitions. The local copies live in `docs/tutorial-assets/walkthrough/`:

```bash
uv run --frozen python docs/tutorial-assets/walkthrough/01_weighted.py
uv run --frozen python docs/tutorial-assets/walkthrough/10_interface.py
```

The first prints the weighted contributions and context `[1.5,1]`. The last reproduces the all-position attention result through the complete actual class and checks its explicit/library paths.

For an exported tutorial package, [the restore script](examples/restore_source.py) rebuilds the captured project described by [the source bundle](code/source-bundle.json). From the tutorial package root:

```bash
python3 examples/restore_source.py restored-project
uv run --project restored-project --frozen python examples/walkthrough/01_weighted.py
```

### Find the owner of a behavior

| Location | Responsibility |
| --- | --- |
| `src/core/attention.py` | Actual attention formula, projections, head layout, masks and interface |
| `src/attention.py` | Runnable component demo; also available as `attention-demo` |
| `src/utils/attention_visuals.py` | Original matrix examples and display helpers |
| `tests/test_attention.py` | Values, gradients, masks, interface and independent PyTorch-reference checks |
| `tests/test_attention_visuals.py` | Known answers for numerical visualization fixtures |
| `notebooks/attention_colab.ipynb` | Source upload/verification, locked environment, demo/tests and downloadable logs |

Runnable scripts live directly under src; reusable groups are `core`, `data`, `training` and `utils`. The tokenizer/data/model/trainer/checkpoint destinations are placeholders for later implementation. An empty named script exiting successfully does not establish that its advertised model action exists.

### Interpret the demo output

The demo uses a fixed seed and prints weighted sums, head calculations, causal weights and cross-attention. Its random-parameter encoder/decoder outputs have shape `(3,5,9)` and cross-attention has `(3,7,9)`. Printed vectors are feature activations, not translated words. Path-agreement checks compare identical computations; they do not measure translation quality or hardware speed.

### Use the existing Colab runner

Upload `notebooks/attention_colab.ipynb` into [Google Colab](https://colab.research.google.com/) and follow its source-archive/manifest upload instructions. Matching local exports live in workspace `.ai/exports/attention-is-all-you-need/`. Its six operational code cells verify files, extract source, install the locked runtime, record versions, run the demo/tests and download logs. The notebook kernel coordinates subprocesses rather than silently substituting Colab's preinstalled PyTorch.

Previous checks exercised those operational cells locally with a simulated Colab file interface. Actual Google-hosted execution and recovery after runtime deletion remain unverified.

### Diagnose a failure at the responsible boundary

If torch cannot import, inspect the selected project environment. If the package cannot import, check the working directory and installation. If dimensions fail, use the tensor contracts in chapters 3 and 10. If a numerical assertion fails, trace the intermediate inputs and operation before changing expected values. Do not treat a successful setup command as a training result.

The [reference record](project/reference-runs.md) identifies author checks, including [independent scalar arithmetic](examples/arithmetic.py) and [actual-component probes](examples/verify_component.py). The [setup learning points](project/learning-points/00-setup.md) and [setup comparisons](project/comparisons/00-setup.md) remain available for environment review. These optional checks support the written route; they are not prerequisite lessons.

[[practice:setup-check]]
'''

def write_pages():
    for name, text in PAGES.items():
        original = (BEFORE / 'lessons' / name).read_text()
        blocks = re.findall(r'^```python\n(.*?)^```\s*$', original, re.M | re.S)
        if '@@PROGRAM@@' in text:
            assert len(blocks) == 1, (name, len(blocks))
            text = text.replace('@@PROGRAM@@', '```python\n' + blocks[0] + '```')
        (COURSE / 'lessons' / name).write_text(text)

if __name__ == '__main__':
    write_pages()
    print(json.dumps({'written_chapters': len(PAGES), 'programs_from_preserved_source': True}))
