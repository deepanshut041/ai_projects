# Let Adam remember gradient direction and size

## Keep direction and squared magnitude

Adam stores two numbers per parameter. The first follows the gradient with its sign. The second follows its squared magnitude, so large recent gradients increase the denominator. We use beta one zero point nine and beta two zero point nine nine nine. Both buffers start at zero. That initialization pulls early averages toward zero, so we correct them using the number of completed updates. The first update is t equals one. A batch of thirty two increments t once, just like a one row update. It is not the count of examples visited. Our implementation uses no weight decay or AMSGrad, and keeps epsilon outside the square root.

## Work through the first hour-8 update

The real hour still begins with gradient negative forty eight point seven four seven three. The first moment stores one tenth of it, about negative four point eight seven four seven. The second stores one thousandth of its square, about two point three seven six three. Dividing by zero point one and zero point zero zero one restores the first gradient and its square. The square root is approximately its absolute magnitude. Their ratio is almost negative one. Rate zero point zero one therefore increases this coordinate by about zero point zero one, much less than the first SGD movement. Across all coordinates, the resulting forecast is one hundred and forty four point zero two four nine rentals.

## Correct the buffers before dividing

The source advances the update counter, then accumulates both buffers. These shortened names correspond to explicit configuration fields. At the second real hour update, the gradient is about minus forty seven point six five. The stored first moment is about minus nine point one five, and the second is about four point six four. At update two, each correction denominator is one minus its beta squared, as shown in the code. Reusing the first step correction would give the wrong calculation. The hour eight coefficient becomes just under two hundredths. Recomputing every coordinate gives a second forecast of about one hundred and forty four point two six rentals. The notes retain the exact values, and the independent scalar trace agrees with the code.

## Reverse a gradient and retain its history

Now change the case. A scalar parameter starts at zero and receives gradient plus two, then minus one, at rate zero point one. The first step takes it to approximately negative zero point one. After the sign reversal, the first moment is still positive zero point zero eight. Correcting that positive average and dividing by a positive magnitude moves the parameter farther down, to negative zero point one two six six three four. Restarting the buffers would instead react to the latest negative gradient alone. The retained history matters. This changing gradient check is more informative than testing only a repeated gradient, because it exposes whether the implementation actually carries state between updates.

## Change the scale of the gradient

Epsilon becomes important for a tiny gradient. With gradient one times ten to the negative ten and rate zero point one, our first corrected update is approximately negative zero point zero zero zero nine nine zero zero nine nine. Putting epsilon inside the square root would produce a different number. A direct arithmetic check protects this convention. At the same rate, Adam moved less than SGD on the opening hour because it normalized gradient magnitude. That is not proof that it is slower in every useful comparison. Our equal allowance pilot selects Adam rate one for the unscaled rental target. Compare measured behavior with declared tuning allowances, and keep the moments, counter and numerical convention in every continuation record.
