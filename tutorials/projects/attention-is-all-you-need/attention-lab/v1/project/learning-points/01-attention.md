# What the attention computation means

## A useful output is a new representation for one query

The eventual translator must let an output position use information from other positions. Attention supplies a differentiable way to form that representation. It compares one query with available keys, turns those scores into a row of weights, and uses that row to combine the corresponding values. With values `[2,4]` and `[6,8]`, equal weights produce `[4,6]`. The same positional weight multiplies every coordinate of one value vector.

This makes attention more specific than “the model focuses on important words.” The result is a calculated vector, and the available positions and score rule determine what can contribute. A large attention weight is only a coefficient. The actual contribution also depends on the value vector, and the final output projection changes the representation again. These synthetic examples do not establish word importance, linguistic alignment or a causal explanation of a trained prediction.

## Q and K decide how to compare; V decides what to carry

Self-attention applies three different affine maps to the same input rows. Equal input origins do not make Q, K and V interchangeable: changing a Q/K parameter changes comparisons, while changing V changes content. Cross-attention uses the same roles but obtains Q from target-side rows and K/V from source-side rows. The separation lets each target query retrieve source information without requiring equal sequence lengths.

In our controlled example, input `[1,0]` becomes query `[1,0]`, key `[1,1]` and value `[2.5,-0.5]`. Those values come from explicitly selected matrices and bias. They are neither identities learned by training nor permanent meanings assigned to two model coordinates. PyTorch's `nn.Linear` stores the transpose of the right-multiplication matrix used in the written equations; knowing that convention matters when checking real code against a diagram.

If Q/K stay fixed and every V row shifts by the same vector, the normalized context shifts by that vector. The weights do not change. If the original input changes instead, Q/K/V may all change. This difference gives us a controlled experiment for understanding the implementation rather than explaining every changed output with a vague change in focus.

## The two matrix products consume different axes

`Q@K.transpose(-2,-1)` consumes head-feature coordinates and produces one score for each query/key pair. Softmax normalizes across the key columns of each query row. `weights@V` then consumes key positions and returns one value-feature vector per query. A mistaken transpose can preserve an element count while giving every matrix cell the wrong meaning.

For one head, three source positions and two queries, weights are `(2,3)` and context has two output rows. This is why cross-attention output length follows the target queries. At the full module boundary, batch and head axes add independent copies of the same calculation. Source masks still have source length, even when target length differs.

Scaling scores by the inverse square root of head width limits the widening of dot-product variation under the usual independent-coordinate reasoning. It does not make all scores comparable in every trained state or normalize vector magnitudes. Stable softmax produces normalized nonnegative weights for nonempty rows. Adding a constant to every score in a row changes neither its distribution nor its context; changing score differences does.

## Heads separate learned feature spaces while retaining all positions

A head receives a subset of the projected feature coordinates, but still compares every permitted position. Several heads can therefore compute different mixtures in parallel. Reshaping does not guarantee a head for grammar or a head for meaning; that would require evidence from a trained model.

The code first reshapes `(B,T,D)` into `(B,T,H,d)`, then transposes to `(B,H,T,d)`. On the way back, it must reverse the position/head order before flattening H and d. Otherwise features from two queries can be grouped into one apparent output row. The output projection mixes the reassembled features. Its bias also explains why zero retrieved context need not mean zero final output.

Changing head count at fixed model width repartitions projected coordinates and changes attention distributions, while the four full-width affine maps retain the same parameter count. That does not make head counts behaviorally equivalent. Loop and vectorized versions are comparable only under equivalent parameter mappings; two fresh random initializations compute different functions.

## Masks restrict influence, and loss masks solve a different problem

Padding removes invalid key columns; causality removes future query/key pairs. With both present, permitted pairs are their intersection. In this module True means allowed. A forbidden score becomes negative infinity before softmax, so it receives zero probability. Replacing it by zero would still permit a positive contribution.

Fully blocked rows cannot use ordinary softmax directly. The explicit helper uses a safe temporary score row, then zeros its normalized weights and context. The final output projection may still add bias. A padded query can read valid keys because the provided mask describes key validity. Later translation training must separately align shifted decoder inputs/labels and exclude unwanted target positions from the loss.

A useful test changes a forbidden source position and asks whether an earlier output changes with dropout off. This tests actual influence, while a triangular-mask picture only shows an intended rule. Both help, but they establish different facts.

## Inspection weights, execution paths and learned behavior are separate observations

`need_weights=True` uses the explicit path and returns weights before attention dropout. During training, the context uses a randomly dropped/rescaled version. The returned row can sum to one while the used row does not. This is why normalized-mixture arguments must state whether dropout is active.

The SDPA path shares the public input/output contract. Evaluation passes probability zero explicitly, and padding plus causality uses the combined mask. Output and gradient parity checks compare matching parameters, inputs, masks, dtype and mode within a tolerance. Calling the SDPA API on CPU does not establish GPU FlashAttention use or performance. The checked local environment and source tests support component correctness on their exercised cases.

## A gradient connects the retrieved representation to a parameter update

The scalar gradient probe adds an illustrative desired context and loss. At q=0, context four versus desired five gives `dL/dq=2`; a small negative query step increases weight on value six and reduces that loss. The chain continues into the query projection's weights and bias. `backward()` computes these derivatives; the optimizer applies updates. A trained translator additionally needs data, complete blocks, task logits/loss, repeated updates and evaluation. Videos and passing checks do not establish the learner's own explanation or those missing task results.

The [ordered tutorial](lessons/01-weighted-information.md) connect each point to the exact source, worked numbers and a changed case. Revisit one section when a shape, mask or gradient result is unclear.
