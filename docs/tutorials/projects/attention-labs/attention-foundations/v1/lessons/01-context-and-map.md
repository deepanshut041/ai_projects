## 1. A map of the attention family

Open two language-model descriptions and you may find **GQA**, **RoPE**, **causal attention**, **MLA**, and **FlashAttention** in the same paragraph. Are these competing ways of doing one operation? Could a model use several together? Before calculating attention, we need a way to ask what each name changes.

![Three model descriptions contain different combinations of attention terms](images/original-slides/lecture-01-slide-03.png)

Read each card vertically. The Llama 3 card combines a head arrangement, a positional method and a visibility rule. DeepSeek-V3 adds a different way of representing keys and values. Kimi Linear combines recurrent attention layers with periodic full-attention layers. These lists describe several decisions inside a model. Learning a separate name without locating its job makes the list look much harder than it is.

This opening lesson builds the map. The next lesson starts the mechanism itself: a translation system reads a sentence, compresses it, and tries to produce another sentence. We will first understand why that fixed compression causes difficulty, then construct attention as a remedy.

### First see the operation the family grows around

![The word sat retrieves a weighted combination, with larger bars for cat and mat](images/original-slides/lecture-01-slide-02.png)

The highlighted word, **sat**, is the position asking for information. Follow its arrows toward the other words, then follow the bars down into the weighted sum. The larger bar for **cat** means more of that position's value is included; **mat** also contributes strongly. Smaller bars still contribute. Attention does not have to choose exactly one word.

These are illustrative weights over the six positions in “The cat sat on the mat”. They are rounded, so the displayed numbers add to 1.00. The returned object is a mixture of numerical value vectors, not the next predicted word. A later model component makes vocabulary predictions. We will open the scoring and normalization that produce these bars in the following lessons.

Now imagine changing one aspect: forbid looking at future words, use several independent mixtures, or compute the same mixture with less GPU memory traffic. All three changes concern attention, but they solve different problems. That is why the family needs a map.

### Seven questions organize the names

![Seven categories: scoring, connectivity, heads and KV, sub-quadratic methods, implementation, position, and domain](images/original-slides/lecture-01-slide-05.png)

Read across the top row, then across the bottom. Each box asks a different design question.

**1. Scoring — how do we measure compatibility?** A position requesting information is represented by a query; available positions have keys. A scoring function compares a query with each key. Additive attention uses a small learned network. Dot-product attention multiplies corresponding coordinates and adds them. Scaled dot-product attention also controls the magnitude of that sum. We will derive those choices rather than treating their names as explanations.

**2. Connectivity — which information paths are available?** A causal rule permits the current and earlier positions while excluding future ones. Bidirectional visibility can permit both directions. Self-attention obtains queries and memory from one sequence; cross-attention retrieves from another sequence. Sequence origin and visibility are related questions, but they are not identical: self-attention may be causal or bidirectional.

**3. Heads and KV — how are retrieval channels arranged and stored?** Multiple heads perform separately parameterized comparisons. MHA, MQA, GQA and MLA arrange or represent these channels differently. During token-by-token generation, a key/value cache retains information from previous positions. Its size depends on these choices as well as sequence length. We will keep MHA and later head/KV techniques together in **Attention Heads**.

**4. Sub-quadratic methods — how do we avoid every possible pair?** For a sequence of length $n$, unrestricted pairwise attention compares $n$ queries with $n$ keys. This gives $n^2$ pairs. Sparse methods retain selected connections. Linear/recurrent methods change the mathematical organization so a state can summarize the past. Their outputs and tradeoffs need to be examined individually.

**5. Implementation — how do we execute the operation efficiently?** FlashAttention reorganizes exact dense attention around GPU memory and tiling. PagedAttention addresses key/value memory organization and sharing. Ring Attention distributes work and communication. These names do not, by themselves, mean that the score formula has changed.

**6. Position — how does order enter the computation?** The same word at the beginning and end of a sentence may play different roles. Sinusoidal encodings, RoPE, ALiBi and context-extension methods address aspects of position. Naming an ordering method does not specify the head arrangement or the visibility rule.

**7. Domain — what structure does an input position have?** Text is a sequence. An image has a spatial grid. A graph has nodes and edges. Image windows, deformable sampling, graph attention and cross-modal retrieval adapt information exchange to these structures.

Use the map as an organizing tool, not a claim that a research method can affect only one box. Sliding windows affect both connectivity and cost; position and caching can interact. A model combines choices across several boxes.

### Four reasons a new method might be useful

The seven questions can be grouped into four practical concerns:

| Concern | Questions to ask | Categories |
| --- | --- | --- |
| What is computed? | How are scores formed? Which query–key pairs are permitted? | Scoring and connectivity |
| What does it cost? | How much state must be stored? How does work grow with length? | Heads/KV and sub-quadratic methods |
| How is it executed? | How much data moves between slow and fast memory? Can work be tiled or distributed? | Implementation |
| What structure is represented? | How does the model know order? Is the input text, an image, a graph or several modalities? | Position and domain |

For example, suppose two implementations return the same attention output, but one moves fewer intermediate arrays through GPU memory. That is an implementation improvement. If another method removes most query–key connections, its mathematical information paths have changed too. Comparing only speed would miss that difference.

Likewise, reducing the number of stored key/value heads can reduce cache memory, but it does not automatically remove the quadratic pairwise work of a full-sequence dense attention layer. Memory size, arithmetic work and measured runtime are different quantities.

### Follow one combination: Llama 3

![A path selects scaled dot-product scores, a causal mask, GQA, RoPE and FlashAttention](images/original-slides/lecture-01-slide-07.png)

Follow the dark choices from left to right. The score is scaled dot-product. The mask is causal. The head arrangement is grouped-query attention. Position is represented with RoPE. The operation can run through FlashAttention kernels.

Nothing about choosing GQA requires replacing the causal rule. Changing the positional method does not automatically change the score into an additive neural network. This is the useful sense in which the choices can be considered separately. They still have to be integrated correctly in a real implementation.

Also, these choices do not specify a complete model. Layer count, hidden width, feed-forward blocks, normalization, vocabulary, training data and objective still matter. Two models can agree on the attention choices and have very different capabilities.

### Compare models by holding the questions fixed

![A seven-row comparison of Llama 3, DeepSeek-V3 and the hybrid Kimi Linear model](images/original-slides/lecture-01-slide-08.png)

Read **across a row** before comparing whole columns. The causal row asks the same question for every model. The heads/KV row distinguishes GQA from MLA. The sub-quadratic row distinguishes an all-full-attention stack from a hybrid containing KDA layers. The domain row says all three examples are text models.

Hybrid models require particular care. Kimi Linear uses a 3:1 pattern of KDA layers and full MLA layers. The full-attention scoring entry does not describe the KDA recurrence. Its NoPE design applies to the MLA layers; the KDA dynamics carry order and recency information. Do not read the compact position entry as “the whole model has no way to represent order”.

The purpose of this comparison is to practice decomposing a specification. It is not a performance ranking, and methods introduced in later DeepSeek work should not be silently assigned to DeepSeek-V3. When you inspect a model, identify its exact version and layer type before filling in the rows.

### What you need before the calculations

You should recognize a vector, matrix multiplication, a neural-network parameter, an MLP, backpropagation, an embedding and softmax. You do not need to know the Transformer architecture already. We will explain each object's role and shape when it enters the mechanism.

If you already know the formula $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$, check whether you can explain its axes, its normalization direction, the source of each matrix and the visibility rule. Being able to reproduce the notation is a weaker check than being able to calculate one row and predict a changed input.

### The full family has eight modules

![An ordered route through foundations, position, KV cache, sparse, linear, implementation, beyond text and the map](images/original-slides/lecture-01-slide-12.png)

The module names are broader than individual lecture numbers. A module can need several lectures. The route is ordered so each problem becomes understandable before its remedy.

**Foundations.** Start with recurrent sequence-to-sequence translation, inspect its fixed-vector bottleneck, then develop additive, dot-product and scaled dot-product scoring. Continue into self-attention, learned projections, masks, cross-attention and multiple heads.

![An encoder squeezes a source sentence into one vector before a decoder produces a translation](images/original-slides/lecture-01-slide-13.png)

The important part is the single middle connection. No matter which target word is being produced, this baseline has to obtain source information through that one fixed summary. Attention will change this interface: retain source-position representations and retrieve a suitable mixture at each output step. Lecture 2 develops the problem; Lecture 3 builds the remedy.

**Position, then KV cache.** First ask how the operation distinguishes token order. Then follow generation one token at a time and ask what can be retained instead of recomputed.

![Each generation step adds another key and value, so the cache grows with the prefix](images/original-slides/lecture-01-slide-14.png)

Read the four steps horizontally. The first stores one K/V pair; later steps retain old entries and append a new pair. The growing memory bar below represents that accumulation. Reuse saves recomputation, but retained memory grows with the prefix. That makes cache layout, head sharing and memory management meaningful topics rather than arbitrary extra names.

**Sparse, then linear.** Next study the cost of a long sequence.

![Full triangular attention, sparse connections and a fixed-size recurrent state provide three different information paths](images/original-slides/lecture-01-slide-15.png)

In the left panel, the allowed causal cells form a triangle: later queries can inspect more earlier keys. Its count is $n(n+1)/2$, which still grows quadratically. Doubling a long sequence roughly quadruples the number of allowed pairs. A 1,000-position sequence has 500,500 causal pairs; 2,000 positions have 2,001,000.

The middle panel drops many of those connections. The right panel routes history through a recurrent state. Sparse attention changes which explicit pairs remain. Recurrent linear methods change how past information is represented and updated. A fixed-size state also imposes capacity tradeoffs; it is not free unlimited memory. A hybrid with some full-attention layers still has those layers' costs.

**Implementation, beyond text, then synthesis.** Once the operation and its costs are clear, study hardware execution, structured inputs and how newer ideas fit the map.

![GPU tiling, a local image window and graph-neighborhood attention](images/original-slides/lecture-01-slide-16.png)

The left panel highlights a tile that can be processed using fast GPU memory. The middle highlights a neighborhood in a two-dimensional image grid. The right shows connections between graph nodes. The colored regions answer different questions: an execution block, a spatial information region and a graph neighborhood. Similar-looking boxes or links need their surrounding problem explained.

The final synthesis returns to the seven questions and places new methods using the problem they change. You do not have to memorize every name now.

### Put the full name list back into the map

The early name cloud becomes easier to read once its entries are organized by the problem they mainly address. Keep this list as a lookup map; you do not need to explain every future mechanism now.

| Main question | Names encountered in the family |
| --- | --- |
| Scoring and related changes to the attention computation | Bahdanau/additive, scaled dot-product, differential, gated, sigmoid |
| Sequence relationship and visibility | Self-attention, cross-attention, causal mask, bidirectional |
| Head and key/value organization | MHA, MQA, GQA, MLA |
| Reducing long-sequence costs | Sliding window, Longformer, BigBird, NSA, MoBA, linear attention, RetNet, GLA, DeltaNet, KDA |
| Execution and memory organization | FlashAttention, PagedAttention, Ring Attention |
| Position and context behavior | Sinusoidal encodings, RoPE, ALiBi, YaRN, attention sinks |
| Input structure and domain | SENet, CBAM, Swin, deformable attention, graph attention/GAT, cross-modal attention |

These placements are a study aid, not exclusive definitions. For example, attention sinks describe a behavior relevant to certain streaming/window strategies, and a sliding window is both a connectivity rule and a cost choice. When a new name appears, ask which mathematical operation, information path or execution constraint it changes before assuming it replaces the entire attention layer.

### Where this tutorial stops, and how to follow it

This **Attention Foundations** route covers Lectures 1–7 in order. The next six lessons explain the recurrent baseline, additive retrieval, alternative scoring and scaling, self-attention, learned projections, and causal masking with dropout. **Attention Heads** then covers Lecture 8, including the explicit-head and packed implementations. GQA and related arrangements belong there when added later. Lecture 9 is excluded.

The broader eight-module map is context for future study, not a claim that all those mechanisms are implemented in this project. The project goal is to build and independently test the basic mechanisms, including a Colab-compatible workflow.

The complete route is useful when your foundations are uneven. A shorter route for reading modern model specifications starts with self-attention, MHA and masks, then continues later into the KV cache, MQA/GQA/MLA, RoPE, FlashAttention and sliding windows with attention sinks. Those later topics are not all present in the current tutorial. Skipping is sensible only when you can explain the dependencies you skip.

[[practice:map-check]]

Before continuing, take one attention name you have encountered and ask: **What problem does it change, and what other choices can coexist with it?** The next lesson turns the fixed-summary diagram into an encoder–decoder process you can trace.

### References

[Vizuara — The Attention Family, Lecture 1](https://www.youtube.com/watch?v=QgnOoQsPc0o) · [Presentation and diagrams](https://lecture1-attention-mechanisms.vercel.app/).

For the hybrid layer and position distinctions: [Kimi Linear technical report, architecture section](https://arxiv.org/html/2510.26692v1#S4).
