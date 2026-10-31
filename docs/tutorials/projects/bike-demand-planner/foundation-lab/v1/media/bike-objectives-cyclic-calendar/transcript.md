# Give the calendar a circle and inspect what it loses

## Keep midnight neighbors close

Hour zero and hour twenty three are neighbors on a clock. A numeric hour column has a jump at midnight, while one hot indicators give no built in notion of neighboring hours. A sine and cosine pair places those hours close together on a circle. Hour eight occupies its own position farther around the clock. The highlighted short connection shows why twenty three and zero are nearby in this representation. That is a useful input geometry, but the prediction function still depends on what the learner can do with those coordinates.

## Calculate the hour-8 coordinates

Divide the hour by its period, twenty four, and multiply by two pi to get an angle. Apply both sine and cosine. At hour eight, the values are about zero point eight six six and minus zero point five. At zero the pair is zero and one. At twenty three it is about minus zero point two five nine and zero point nine six six, close to the midnight point. Weekday and month use their own periods. January is indexed at zero in our declared transformation.

## Build eight inputs without reading outcomes

The source recovers calendar indices from the frozen one hot vector and computes each pair. It reads no outcomes. Three pairs make six columns; holiday and working day add two, giving eight inputs. We retain their exact order and fit any standardization on training only. The pair still distinguishes the discrete hour positions. What changes is how a linear function can combine them. This representation is compared with the original one hot model using the same training solve, no penalty and the same later rows.

## A linear pair gives one smooth hourly wave

A linear model can multiply the sine and cosine by one coefficient each and add an intercept. That combination produces one sinusoidal wave for the hourly contribution. It cannot freely assign twenty four independent hour effects as one hot columns can. Two sharp commuting peaks are therefore difficult to express with this one pair. The calendar position remains encoded, but linear expressive freedom has changed. Adding harmonics or using another nonlinear learner could change that limitation; either would be a new declared comparison, not something already implemented here.

## Read capacity through fit and validation

The cyclic training half squared error is about four thousand, eight hundred and ten, higher than the one hot reference near three thousand, one hundred and sixty eight. Raw validation absolute error is slightly lower for cyclic features, but after consistent clipping it is worse. Root mean squared error is also worse. These are stable training solves, so unfinished gradient optimization does not explain the structural difference. A smaller representation has not supplied a universally stronger predictor. Keep the representation, output policy and metric beside each result.

## Return to the busy Monday hour

At the later Monday eight oclock hour, the cyclic model predicts only about fifty two rentals against four hundred and twenty. The one hot reference predicts about one hundred and seventy five. The clock picture is sensible, but the resulting linear function is too restrictive for this example. That does not make circular encoding useless for every learner or problem. It tells us what this particular linear pair can express under the frozen contract. We next add an explicit hour by working day interaction, which answers a different structural question.
