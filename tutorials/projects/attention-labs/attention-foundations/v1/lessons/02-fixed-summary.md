## 2. Why one fixed summary can be difficult to use

The previous lesson located attention inside a translation system. To see why it helps, first build a system whose output side can access only one final input summary. This is a useful historical starting point, not a claim that every recurrent model necessarily loses its earliest input.

### Follow the input into recurrent state

Let the three source tokens have embeddings $x_1,x_2,x_3$. An embedding is a vector selected by token identity. A recurrent encoder updates a state after each token:

$$h_t=\tanh(x_tW_x+h_{t-1}W_h+b_h),\qquad h_0=0.$$

Here rows represent vectors: $x_t$ has width $d_e$, $h_t$ has width $d_h$, $W_x$ has shape $d_e\times d_h$, and $W_h$ has shape $d_h\times d_h$. The two projected rows can be added because they have the same width. The elementwise $\tanh$ supplies a nonlinearity. The matrices are reused at every step; $h_1,h_2,h_3$ are different computed states, not three independently learned parameter sets.

An encoder can use different widths for embeddings and hidden states. The fact that both are vectors does not require equal dimensions or make them interchangeable. After the third update, $h_3$ depends on all three inputs through the recurrence.

The input embedding can be understood as a table lookup. If token ID 4 selects row 4 of an embedding table $E\in\mathbb R^{V\times d_e}$, then $x_t=E[4]$; $V$ is the input vocabulary size. A new model can learn this table jointly with its other parameters, or a system can initialize or freeze it using pretrained vectors. Similar directions can emerge from training, but the token ID itself contains no geometric meaning. Our later binary coordinates are deliberately chosen arithmetic inputs, not entries from a fitted language model.

![A fixed source-to-decoder handoff](images/02-fixed-summary.svg)

In this baseline, the decoder receives $c=h_3$. A decoder state can be initialized from $c$, or a decoder can also condition on $c$ at each update. Those are implementation choices; our limitation is that the input information reaching the decoder must pass through that same fixed-width representation.

### Quantify one possible fading effect

Use a scalar recurrence to isolate the effect of repeated state updates:

$$h_t=0.5h_{t-1}+x_t.$$

For inputs `[1,0,2]`, the states are $h_1=1$, $h_2=0.5$, and $h_3=2.25$. Expanding the last state reveals its dependencies:

$$h_3=0.25x_1+0.5x_2+x_3.$$

Increasing only $x_1$ by one raises the final summary by $0.25$. With $T$ tokens, the corresponding coefficient is $0.5^{T-1}$. This toy recurrence demonstrates an attenuating path; it is not a fitted RNN, a universal formula for memory, or an information-theoretic impossibility proof. Learned recurrent matrices and gates can behave differently. Long sequences make learning and preserving useful summaries challenging, but there is no universal twenty-word failure threshold.

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

### Follow the summary into output tokens

The decoder begins with a start token, usually written `<BOS>`. At step $t$, it combines its previous state with the embedding of the previous output token. A conventional output head computes

$$\ell_t=s_tW_o+b_o,\qquad p(y_t\mid y_{<t},c)=\operatorname{softmax}(\ell_t).$$

There is one logit in $\ell_t$ for each vocabulary entry. Selecting the next token from this distribution differs from finding the nearest embedding vector. The chosen token is then supplied to the next decoding step. An `<EOS>` token can end generation, with a maximum-length safeguard for a model that never emits it.

The source length and output length therefore need not match. Encoding three tokens does not require decoding three tokens. During teacher-forced training, the previous **reference** output token is supplied instead of the model's last sampled token; the vocabulary prediction at a position still targets the next reference token.

### Change the interface rather than just enlarging the summary

The encoder has already computed states for intermediate positions. Retaining them makes a different interface possible: give the decoder access to $H=[h_1;h_2;h_3]$, then construct a context vector according to what the current output step needs. Storage now grows with the source length, instead of requiring every retrieval to use only $h_3$.

Attention will still return a finite-width context vector at a step. The change is that it can build **a different context for each step from the retained memory**. It removes this particular fixed handoff, not every limitation of representation, training, or translation.

[[practice:summary-check]]

### The original lecture and slides

[Vizuara Lecture 2](https://www.youtube.com/watch?v=IG2gl6sKj_c) develops the recurrent encoder–decoder motivation. In the original presentation, slides 8–13 show the cell, states and handoff; slides 15–18 discuss the limitations. Our scalar probe makes one of those limitations measurable, while the vocabulary-head explanation makes the prediction interface explicit.

```presentation
{"title":"Vizuara · Lecture 2 · Original presentation","url":"https://lecture2-attention-mechanisms.vercel.app/#/1"}
```
