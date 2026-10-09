## Start with a picture you can explain

This is a reference-design study before P01-M1. Draw the boards yourself and predict the small examples before running them. The GPT repository, full decoder and training run are still pending. Reading this lesson does not establish mastery.

### Find a useful starting point

The route below is proposed; no learner answers have been observed. We assume familiarity with C++/Python arrays and indices, but tensor shapes, softmax and target alignment remain unchecked. Try this entry question before reading its answer. If it is easy, continue to the changed-value and zero-score checks; if it is unfamiliar, begin with the three-position trace in the next chapter.

[[practice:visible-prefix]]

Our task is to predict a later token from a permitted prefix. That task premise motivates restricting which positions can contribute. We will derive zero future weight, calculate a small prefix mixture, then test whether changing the future can affect an earlier output. The arrows on this board are concept prerequisites, not model execution or proof of learner progress.

[[diagram:learning-path]]

After the causality trace, align each prediction with its next-token target. Visibility controls what information is available; target alignment specifies the question the model is trained to answer. Finally draw the source-specific GPT-1 block and distinguish growing cache from other storage. You can inspect the architecture boards now and return to them after the trace; a live session can work through one question at a time.

The inspected [original implementation](source:gpt-code) supplies the normalization order and target alignment. The [paper](source:gpt-paper) describes the architecture and training setup. Our future tiny model uses byte/special IDs rather than the published tokenizer; that is an educational simplification.

[[diagram:decoder]]

At each token, add its learned position vector to its token vector. Feed the sequence through repeated causal blocks, then turn each resulting vector into vocabulary scores. Positions supply order; the mask controls visibility.

[[diagram:postnorm]]

Follow the two residual branches explicitly. A block keeps the outer batch/time/width shape even though its feed-forward intermediate expands. Do not replace this reference's normalization order with a remembered diagram of another model.

[[practice:draw-block]]
