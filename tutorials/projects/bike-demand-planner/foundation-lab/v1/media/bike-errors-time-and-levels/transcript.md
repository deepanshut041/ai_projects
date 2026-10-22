# Keep time order when the demand level changes

## The errors have a position in time

January and June do not merely contain different error magnitudes. They occupy different positions in the forecast timeline. The daily plot keeps that order and shows mostly larger negative residuals later in the validation period. We summarize each date using its actual recorded hours, then form Monday starting weekly groups. There are twenty seven such groups, with counts ranging from twenty three to one hundred and sixty eight. Partial boundary weeks remain partial. We have not shuffled observations or invented missing targets to make every day and week look equally complete.

## Only adjacent hours enter the correlation

The validation window contains ten unrecorded hourly slots. A row after a gap must not be treated as exactly one hour after the row before it. The pair builder therefore compares timestamps, retaining only differences of one hour. It finds four thousand, three hundred and forty six consecutive hour pairs. For these pairs, the residual correlation is about zero point eight six eight. The full function guards against too few pairs and zero variation before computing correlation. Missing slots remain missing; inserting zero outcomes would alter both the evaluation population and its apparent time structure.

## Describe dependence without inventing an interval

A strong positive correlation says neighboring residuals tend to move together in these observed pairs. It does not prove a cause or guarantee persistence. Weekly mean absolute error ranges from thirty two point seven five to one hundred and fifty seven point two six on the observed weeks. That range is descriptive, not a confidence interval for next week. The earlier forest seed standard deviation near zero point zero five five has another meaning. It measures fitting variation on the same validation dates. Repeating seeds does not produce new future periods. We retain those distinctions instead of reporting an unsupported independent observation interval.

## Compare historical cells at the later calendar mix

To inspect the level difference, first match month, hour and working day cells between the two years. Within each cell, calculate a historical mean from January through June twenty eleven training targets. Weight that old cell mean by the number of later validation rows belonging to the cell. This asks what the old within cell levels look like under the later calendar composition. All four thousand, three hundred and fifty seven validation observations find a historical cell, with none dropped as unmatched. The weighting changes composition for this diagnostic, while the previously fitted candidate itself remains unchanged.

## Work through the observed level comparison

The reweighted historical cell means average one hundred and twenty one point three one rentals per hour. The later observed mean is two hundred and fifteen point two zero. Their difference is about ninety three point eight nine. This is distinct from the selected group predictions mean of one hundred and forty two point eight six, because the diagnostic conditions on month too and uses only matching first half historical cells. It is not a newly ranked forecasting candidate. Its role is to make an observed level difference easier to inspect after outcomes have become available.

## Keep the explanation within the evidence

Matching month, hour and day type makes a calendar composition only explanation less adequate, because a substantial difference remains within those shared cells. It does not identify which real world cause produced that difference. Demand growth, unmodelled weather or other changes could contribute. We have not measured their separate effects or carried out a causal intervention. If a different later mix were observed, its weights and diagnostic could change even with the same historical means. That is why the weights, unmatched count and scope are saved, and why this picture does not silently authorize refitting or changing the frozen split.
