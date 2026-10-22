# Follow the same hour through SGD and momentum

## Return to one real training hour

Let us follow the same real hour through two update rules. Monday, the third of January, twenty eleven, at eight, has target one hundred and fifty four rentals. Under our convention the request is issued at seven for the eight to nine interval. Its outcome stays separate from calendar inputs. We initialize zero weights and a bias of one hundred and forty three point seven nine, the training mean. These statistics use the completed twenty eleven window. We are inspecting a training calculation, rather than claiming they existed on January third. Repeat this one row twice at rate zero point zero one. The later dataset experiment uses all training rows.

## Calculate the direction before moving

The residual is prediction minus target, so it is negative ten point two zero five five. Our forecast is too low. The standardized hour eight feature is four point seven seven six five. Multiplying it by the residual gives that coordinate gradient. The result is negative forty eight point seven four seven three. Subtracting a negative gradient increases the coefficient. The first weight becomes zero point four eight seven four seven three. The bias also increases, using the residual as its gradient. Every centered feature participates. Recomputing the complete dot product gives one hundred and forty eight point three seven four three rentals, not merely the contribution from hour eight.

## The remembered direction changes step two

Plain SGD uses the current gradient as its direction. Our momentum convention adds the current gradient to ninety percent of the previous velocity. That buffer starts at zero, so the first updates agree exactly. There is no dampening or Nesterov step. On the second update we recalculate the derivative at the new parameters. The hour eight gradient is now negative twenty six point eight seven one four. Momentum remembers part of the larger first gradient. Its new velocity is zero point nine times negative forty eight point seven four seven three, plus negative twenty six point eight seven one four. The total is negative seventy point seven four four zero, before multiplying by the rate.

## Compare the complete forecasts

SGD reaches hour eight weight zero point seven five six two after the second update. Its complete forecast is one hundred and fifty point eight nine eight nine. Momentum reaches weight one point one nine four nine and forecasts one hundred and fifty five point zero two zero eight. It has crossed the target, even though both rules used the same current derivative. The difference comes from remembered direction. The independently calculated scalar trace agrees with the runnable probe to floating point precision. This checks arithmetic rather than ranking the rules. Two repeated observations cannot establish which method forecasts a later dataset better. We measure that separately with fixed membership and paired seeds.

## Change the memory, then predict the effect

Suppose the momentum coefficient becomes zero. The old buffer disappears from the equation, and each direction equals the current gradient. We recover SGD. With a persistent gradient and coefficient zero point nine, repeated additions approach ten times that gradient. This explains why matching numerical rates does not match movement. When the gradient changes sign, the old buffer can initially resist the new direction. That can carry us past a one row target. In the full dataset pilot, momentum selects a smaller base rate than SGD. We do not assume a universal best rate from a familiar default. The useful question is what state determines this step, and whether the measured result improves under the same work and information boundaries.
