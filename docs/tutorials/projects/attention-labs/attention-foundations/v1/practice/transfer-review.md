# Transfer review: change the conditions

Try these with the worked lessons closed. The aim is to identify which part of the computation changes; a finished program alone does not supply that explanation.

## The query contribution cancels

Replace the additive scorer by $e_j=u^T(W_ss+W_hh_j)$. Change the query while leaving memory and parameters fixed. Predict whether the attention weights change, then justify your answer algebraically.

<details><summary>Worked reasoning</summary>

The query contributes the common scalar $u^TW_ss$ to every score. Softmax cancels a common additive shift, so the distribution does not change. A nonlinear interaction, such as the tanh in the actual worked scorer, can make the query alter relative source scores.

</details>

## A seemingly harmless padded row

Append a zero key and zero value to an unmasked retrieval. Does the context necessarily stay the same because the extra value is zero?

<details><summary>Worked reasoning</summary>

No. The extra key gives score zero, whose exponential is one before the common stabilization factor. It changes the softmax denominator, reducing the other weights, even though its own value contributes zero. Mark the padded key invalid explicitly. Query validity and loss exclusion are separate concerns.

</details>

## The correct triangle with the wrong target

A training loop supplies token $y_t$ at input position $t$, permits the diagonal, and asks that same position to predict $y_t$. Is the causal mask enough to ensure an honest next-token objective?

<details><summary>Worked reasoning</summary>

No. The answer is already present at the allowed current input position. The input/target shift must match the intended conditional prediction. In the worked next-token setup, the output from position $t$ targets $y_{t+1}$; the mask then prevents access to that future input token.

</details>
