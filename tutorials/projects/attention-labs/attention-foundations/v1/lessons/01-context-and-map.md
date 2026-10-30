## 1. What information does a token need?

Consider a translation system reading “Quiet cats nap” and producing “Des chats calmes dorment”. The output is a new sequence, with its own length and word order. A representation of “cats” alone cannot specify which adjective describes these cats, or which action they perform. The system must combine information from several input positions before predicting output tokens.

There are three jobs in that description. First, turn input tokens into numerical representations. Second, let the representation used for a prediction retrieve useful information from other positions. Third, turn the resulting features into scores for possible output tokens. Attention does the second job. It does not tokenize the sentence, create a vocabulary, or by itself constitute a trained translator.

![Where attention contributes information](images/01-context-map.svg)

The arrows show information flowing toward a prediction. In later lessons we open the middle box: a numerical request is compared with available source representations, comparisons determine mixing weights, and those weights combine the available content. We will inspect every intermediate number. The sentence gives the task a purpose; the small coordinates used for arithmetic below are hand-chosen examples, not measured meanings of English words.

### The route through this folder

The next lesson first removes attention and shows what happens when a decoder receives one fixed summary. Lesson 3 replaces that fixed handoff with a different mixture for every output step. Lesson 4 studies how to compute the comparisons. Lessons 5 and 6 apply the mechanism within one sequence and give it trainable projections. Lesson 7 restricts the information available during next-token prediction. At that point we have one complete, inspectable attention head.

Multi-head attention has its own `attention-heads` folder. It builds on the single-head result, and will later house GQA and related arrangements. There are no separate basic-type project folders to navigate between. Lecture 9, on positional embeddings, is deliberately excluded from this course.

### Attention names answer different questions

![A map separating attention scoring, connectivity, heads and other design choices](images/original-slides/lecture-01-slide-05.png)

*Original diagram: [Vizuara, Lecture 1, slide 5](https://lecture1-attention-mechanisms.vercel.app/#/5). Captured from the linked presentation website. Open the image for full size.*

Read the boxes as different questions about one model. A causal, multi-head layer can still use scaled dot-product scoring: the choices combine. The first two boxes guide this foundations route; the head/KV box leads to the separate Attention Heads route. Our table below spells out these questions without assuming that every research method fits only one box.

Suppose a layer uses scaled dot-product scores, a causal mask, and multiple heads. Those descriptions can all be true simultaneously: they describe different choices inside the same layer. Treat the following as questions to ask about a design, not a rule that every published method belongs to exactly one box.

| Question | An example answer | What that answer changes |
| --- | --- | --- |
| How is a query compared with a key? | Additive or scaled dot-product scoring | The numerical compatibility function |
| Where do query and memory representations come from? | Self-attention or cross-attention | Which sequences supply the inputs |
| Which query–key pairs are allowed? | Causal or unrestricted visibility | The support of the attention distribution |
| How are query heads and key/value heads arranged? | MHA now; GQA later | Independent projections and memory sharing |
| What calculation and storage are used? | Dense attention or an alternative method | The mathematical operation or its execution cost |
| How does the model represent order? | A positional method | Information supplied about token positions |
| What kind of input is represented? | Text, images, or another domain | The meaning of a position and its features |

For example, cross-attention can be implemented with multiple heads. Causal masking can be used with MHA or GQA. A faster implementation can preserve the same mathematical outputs. Some methods span several of these concerns, and some modern attention alternatives do not use the exact softmax mixture developed here. Naming a method is therefore the start of its explanation, not a complete specification.

### One concrete operation before the symbols

Imagine three available value vectors: `[1,0]`, `[0,1]`, and `[1,1]`. If a query assigns weights `[0.2,0.3,0.5]`, the returned content is

$$
0.2[1,0]+0.3[0,1]+0.5[1,1]=[0.7,0.8].
$$

The coordinates are features, not word probabilities. The three weights form a distribution over **source positions**. Later, a separate prediction layer can turn the two returned features into a different distribution over **vocabulary tokens**. Confusing those two distributions makes attention appear to predict words directly.

If the next output step changes the weights to `[0.6,0.1,0.3]`, the same stored vectors return `[0.9,0.4]`. Attention can change what is retrieved without recomputing the stored content. Our next question is where those source vectors come from, and why selecting them dynamically helps.

[[practice:map-check]]

For additional changed-condition problems, use the optional [transfer review](practice/transfer-review.md) after the worked route.

### The original lecture and slides

The [caption provenance](project/source-provenance.json) records the English subtitle sources and hashes used in the lecture review.

This lesson corresponds to [Vizuara Lecture 1](https://www.youtube.com/watch?v=QgnOoQsPc0o). The linked presentation gives the series-wide map; this tutorial develops only the eight included lectures, with MHA separated as requested. Our examples, diagrams and prose are independently authored.

```presentation
{"title":"Vizuara · Lecture 1 · Original presentation","url":"https://lecture1-attention-mechanisms.vercel.app/#/1"}
```

For a first session, stop when you can distinguish source-position weights from vocabulary probabilities and locate attention between representation and prediction. The [implementation record](project/implementation.md), [readiness record](project/readiness.md), [source review](research.md), [lecture coverage](project/source-map.md), and [learning points](project/learning-points.md) are available for returning to the course; they are not a second reading route.

The [inline visual source record](project/visual-provenance.json) identifies the original slide and captured diagram for each presentation image used in this route.
