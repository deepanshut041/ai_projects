# Give working days their own hourly shape

## Find the relationship the additive model lacks

The additive calendar model gives each hour its own effect and adds one shared working day adjustment. Holding the other inputs fixed, that day type adjustment is the same at eight in the morning and two at night. Our observations suggest those differences may need distinct hourly shapes. A new optimizer or another scale cannot create a product that is absent from the model. We make that relationship explicit in the representation and compare it under the same training least squares solver.

## Add one active product for the real Monday

We retain every original calendar column and append twenty four products. Each multiplies one hour indicator by the working day flag. For the real Monday at eight, the hour eight product is one, and the other new products are zero. Its declared position is index fifty three, after the original forty five columns. A nonworking hour has zero in every appended product. The original hour indicators still remain available. The feature construction uses calendar values only; observed rental outcomes do not enter it.

## Construct the products before fitting

The first twenty four columns are hour indicators. The last working day flag is selected as a column, preserving broadcasting across rows. Multiplication creates the twenty four products, and column stacking appends them to the original design. We then fit training only scales and solve the same unpenalized squared error objective. The saved order matters because valid dimensions alone cannot reveal a misplaced coefficient. No weather or demand history was introduced, and the split membership and issue time contract remain the same.

## Let the contrast depend on the hour

In raw coordinates, the hourly contribution now contains an hour effect plus an hour specific product coefficient when the day flag is one. The working day contrast includes the shared flag coefficient and that hour specific term. It can therefore differ at eight and two. This is additional representational capacity, rather than a faster route through the old family. The design still contains dependent columns, so individual coefficients are not unique causal effects. We assess the fitted function and its later predictions, not a story about one isolated weight.

## The added structure changes the measured function

Training half squared error falls from about three thousand, one hundred and sixty eight to one thousand, nine hundred and thirty two. Raw validation absolute error improves to about one hundred and eight point six nine. The nonnegative version reaches one hundred point five two. Because both representations use stable training solves, this supports a concrete structural comparison. Extra flexibility does not guarantee every validation row improves or that counts become nonnegative. The complete raw and clipped predictions remain saved beside the model and its feature schema.

## Compare both Monday hours and the baseline

At the busy Monday hour, the forecast rises from about one hundred and seventy five to two hundred and sixty, closer to target four hundred and twenty. At the quiet target five, the raw forecast becomes more negative, around minus eighty four. The group baseline still has lower overall absolute error at ninety one point five seven. The expanded model can represent groups, but its extra fitted calendar effects can generalize less well to the later period. This observation does not identify a single cause of the remaining gap. We retain the stronger reference instead of selecting the newest model by default.
