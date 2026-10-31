# Learning points

It mixes head-feature channels separately at each token. Token-to-token mixing already occurred inside each attention head. In our first row, [1,1] multiplied by the declared W_O becomes [0.5,1.5]. Multiple heads allow different distributions but do not guarantee unique linguistic roles.

Its last two axes are heads and features. That product compares heads for each token, producing [B,T,H,H]. Transposing to [B,H,T,d_h] makes the contraction produce [B,H,T,T], one token-by-token score matrix per head and sequence. Merging outputs also requires restoring token/head order, not merely matching element counts.

No. Our packed MHA preserves separate Q/K/V parameters per head and H_q=H_kv. GQA deliberately shares key/value heads among query heads, which needs an explicit grouping rule and corresponding correctness checks. Neither packing nor grouping automatically removes the sequence-length cost of the score calculation.
