## 2. Before attention: build the translation machine

The seven-category map tells us how attention methods differ. Now step back to a machine that has no attention at all. We will open it in three passes: how it reads and writes a sentence, what crosses between those two jobs, and why that handoff becomes difficult for long sentences.

![The family route begins with foundations before position, caching and efficiency](images/original-slides/lecture-02-slide-02.png)

We are at the first stop, **foundations**. Before deciding how to score a source position, we need to see why a decoder should be able to consult source positions in the first place.

![Scoring is the first attention category we will reach, after understanding the baseline](images/original-slides/lecture-02-slide-03.png)

The highlighted scoring box is the next destination. This lesson establishes the problem that gives that score a purpose.

### A sentence goes in; another sentence comes out

![This is a car enters a sequence-to-sequence model and C'est une voiture leaves it](images/original-slides/lecture-02-slide-06.png)

Treat each displayed word as one token for this small example. Four English tokens enter; three French tokens leave. Translation does not require equal lengths. Nor is it generally a position-by-position substitution: translating “European Economic Area” into French changes the adjective order. The machine must represent relationships and generate an appropriate output sequence.

Real tokenizers may split a word into several subword tokens. That changes the sequence positions, but not the need to read one sequence and produce another. We use word-sized tokens here to keep the diagrams readable.

### Words enter as numerical vectors

![A two-dimensional illustration places fruit embeddings together and animal embeddings together](images/original-slides/lecture-02-slide-07.png)

Each dot is the endpoint of a vector from the origin. The axes show two illustrative coordinates, not two named properties such as “fruitness” and “animalness”. Apple, orange and mango occupy one region; cat, dog and cow occupy another. The purpose is to visualize how learned representations can reflect useful similarities.

For nonzero vectors $u$ and $v$, cosine similarity is

$$\cos\theta=\frac{u\cdot v}{\lVert u\rVert\lVert v\rVert}.$$

A small angle gives a cosine near one. The geometry depends on how the vectors were trained; semantic similarity is not guaranteed merely by assigning token IDs. An embedding table $E\in\mathbb R^{V\times d_e}$ stores one row per vocabulary entry. If “car” has ID 4, its input vector is $x_4=E[4]$. Such a table can be learned with the translation system or initialized from pretrained embeddings.

The picture is two-dimensional so we can draw it. Real embeddings can have hundreds or thousands of coordinates. Their dimension is an architectural choice, not the sentence length.

### Open one encoder cell before repeating it

![The previous state h3 and the embedding of car enter one recurrent cell, producing h4](images/original-slides/lecture-02-slide-08.png)

Follow the two inputs. The arrow from the left brings $h_3$, a state computed after reading “This is a”. The lower arrow brings the embedding of **car**. The cell combines them into $h_4$, the updated state after “This is a car”.

The embedding represents the current token; the hidden state represents the processed prefix. Both are vectors, but their jobs differ. Their widths can differ too. A simple recurrent update, using row vectors, is

$$h_t=\tanh(x_tW_x+h_{t-1}W_h+b_h).$$

If $x_t$ has width $d_e$ and $h_t$ has width $d_h$, then $W_x$ has shape $d_e\times d_h$, $W_h$ has shape $d_h\times d_h$, and $b_h$ has width $d_h$. The projected inputs have matching widths before addition. The elementwise $\tanh$ introduces a nonlinearity.

### Unroll the same cell through the sentence

![One recurrent encoder is applied at four time steps to This, is, a and car](images/original-slides/lecture-02-slide-09.png)

Start with $h_0$, chosen as zero in this example. Combine it with “This” to obtain $h_1$. Combine $h_1$ with “is” to obtain $h_2$. Continue through “a” and “car” to obtain $h_3$ and $h_4$.

The four boxes are four applications of **the same encoder cell**. Its matrices are reused; the states change. Drawing the time steps side by side makes the dependencies visible without drawing a looping arrow through one box. It does not create four independently trained RNNs.

Zero initialization is a convention, not the definition of an RNN. A model can use another specified initial state. Likewise, LSTM and GRU cells use more elaborate state updates. The important dependency here is that information from an early input reaches $h_4$ through successive state updates.

### Name the handoff: the context vector

In this baseline, the context is the final encoder state:

$$c=h_4.$$

If the source has fifty tokens, the context becomes $h_{50}$, but it still has width $d_h$. Sentence length changes the number of updates, not the number of coordinates in the final state.

![The complete encoder–context–decoder machine has one bridge between the two recurrent chains](images/original-slides/lecture-02-slide-12.png)

Trace the left chain, the single bridge, then the right chain. The decoder does not receive the original four token embeddings or the separate intermediate encoder states. All source information available through this interface crosses in $c$.

The encoder may retain intermediate activations internally for training. Saying they are “thrown away” means they are absent from this decoder's retrieval interface, not that automatic differentiation must erase them from memory.

### Open the decoder and follow the feedback arrows

![A decoder generates C'est, une and voiture while passing its state and previous token forward](images/original-slides/lecture-02-slide-13.png)

The context starts the decoder. Its recurrent states are called $s_1,s_2,s_3$ to distinguish them from encoder states $h_1,\ldots,h_4$. At each step it produces features for a vocabulary prediction and a state for the next step. The dashed feedback arrow supplies the previously generated token's embedding to the following update.

A practical implementation starts with a `<BOS>` token and initializes the decoder state from $c$, possibly through a learned projection. It may also condition every update on $c$. A vocabulary head then computes

$$\ell_t=s_tW_o+b_o,\qquad p(y_t\mid y_{<t},c)=\operatorname{softmax}(\ell_t).$$

There is one logit in $\ell_t$ for each output-vocabulary entry. The diagram's “nearest word” label is an intuitive shorthand; conventional translation decoders use learned vocabulary logits and a selection/search procedure, not a mandatory nearest-embedding lookup.

After producing “C'est”, feed its embedding into the next update to predict “une”; repeat to predict “voiture”. An `<EOS>` prediction ends the sequence, with a maximum-length safeguard. During teacher-forced training, the previous reference token is supplied instead of the model's last generated token. In either case, the output length is determined by the decoding process and need not equal the source length.

### Difficulty 1: a longer sentence gets the same-size handoff

![Six words and fifty words must cross through context vectors of the same width](images/original-slides/lecture-02-slide-15.png)

Compare the two input rows and the two boxes on the right. The second source has many more positions, yet its context has the same width. The problem is the burden placed on that shared representation: it must preserve whatever every later translation step will need, without allowing another direct consultation of the source.

For an 800-coordinate context, dividing 800 by eight words gives 100, while dividing by forty gives 20. This arithmetic is only a capacity analogy. Coordinates are not allocated to individual words, and the argument is not a theorem that a fixed-width real-valued vector cannot encode a longer sequence. The practical issue is learning a robust, useful summary through a finite computation and finite precision.

### Difficulty 2: repeated updates can weaken an early contribution

![Earlier words fade while later words become darker along the recurrent path](images/original-slides/lecture-02-slide-16.png)

The color progression illustrates recency bias: repeatedly updating a summary can make earlier information harder to preserve or retrieve. It is not a measured activation trace, and every real RNN does not necessarily forget at the same rate. Gates and learned matrices can preserve information much better than a simple scalar recurrence.

We can measure one specific fading path by choosing

$$h_t=0.5h_{t-1}+x_t,\qquad h_0=0.$$

For inputs `[1,0,2]`, the successive states are $1$, $0.5$ and $2.25$. Expand the last update:

$$h_3=0.25x_1+0.5x_2+x_3.$$

An increase of one in the first input reaches the final summary as $0.25$ because it passes through two multiplications by $0.5$. With $T$ inputs its coefficient would be $0.5^{T-1}$. This isolates a dependency path; it is not a universal memory law.

<!-- example:02_fixed_summary.py:start -->

Complete file: [02_fixed_summary.py](examples/02_fixed_summary.py).

```python
import json


def encode(inputs, retention=0.5):
    state = 0.0
    states = []
    for value in inputs:
        state = retention * state + value
        states.append(state)
    return states


if __name__ == '__main__':
    original = encode([1.0, 0.0, 2.0])
    changed = encode([2.0, 0.0, 2.0])
    assert original == [1.0, 0.5, 2.25]
    assert changed[-1] - original[-1] == 0.25
    print(json.dumps({'states': original, 'changed_final': changed[-1],
                      'first_input_sensitivity': 0.5 ** 2}))
```

<!-- example:02_fixed_summary.py:end -->

Run from this lesson package, `ai_projects/docs/tutorials/projects/attention-labs/attention-foundations/v1/`:

```bash
python3 examples/02_fixed_summary.py
```

The output records states `[1.0,0.5,2.25]`, a changed final state `2.5`, and sensitivity `0.25`. These are computed properties of the declared recurrence. If the last two values differ by one rather than a quarter, the program is not applying the retention factor at both intervening updates.

### Difficulty 3: the relevant word may be far away

![Teacher and smiled are connected across an intervening clause, while students is nearby](images/original-slides/lecture-02-slide-17.png)

Read the sentence: “The teacher who was teaching a difficult concept to the students, smiled.” Who smiled? **The teacher**. The nearby noun “students” is part of the intervening clause. The long upper arrow marks the relation that matters; the short lower arrow marks the misleading proximity.

The decoder needs information about the early subject when translating the later verb. In the fixed-summary interface, that relation must survive the encoder's intervening updates and be usable from the same context used for every other output word. LSTMs and GRUs help with long dependencies, but the fixed source handoff remains a separate limitation. No universal rule says training can never overcome a particular dependency.

### Read the translation-quality graph carefully

![A schematic quality-versus-length graph contrasts a declining fixed-summary curve with a steadier attention curve](images/original-slides/lecture-02-slide-18.png)

The horizontal axis is source sentence length. The vertical axis is translation quality. The gray line illustrates a fixed-summary system deteriorating as sentences get longer; the purple line illustrates why allowing source retrieval was valuable.

The dotted line near twenty words emphasizes the historical motivation. It is not a universal failure boundary. This drawing has no dataset, metric values or error bars from which to infer an exact threshold or constant quality on arbitrarily long inputs. Bahdanau and colleagues reported better behavior on longer sentences for their evaluated attention system; that result does not make every attention model immune to length problems.

The three difficulties share one interface: every decoder step depends on the same source summary. Context vectors themselves are useful. The next design changes **how the context is constructed**, not whether a decoder is allowed to use a vector.

### Keep the encoder states and let the decoder look back

![While writing chat, the decoder assigns 0.75 of its retrieval weight to cat and smaller weights to the other source positions](images/original-slides/lecture-02-slide-20.png)

The source is now “The cat sat today”. While producing the French word **chat**, the decoder places its largest illustrative weight, $0.75$, on **cat**. The other weights are $0.05$, $0.15$ and $0.05$; together they sum to one. At a different output step, the weights can change.

To make this possible, retain the source states $H=[h_1;\ldots;h_n]$. Instead of receiving only the final state, the decoder builds a context from these retained vectors according to its current need. The result at one step is still a finite-width vector. The difference is that it can be a **different vector at each output step**.

[[practice:summary-check]]

You can now trace the input embeddings, encoder states, fixed context, decoder state and vocabulary prediction separately. The remaining question is precise: **How does the current decoder state produce those source-position weights?** The next lesson builds that calculation.

### References

[Vizuara — The Attention Family, Lecture 2](https://www.youtube.com/watch?v=IG2gl6sKj_c) · [Presentation and diagrams](https://lecture2-attention-mechanisms.vercel.app/).

[Bahdanau, Cho and Bengio — Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/html/1409.0473v7), encoder–decoder background and length-dependent results.
