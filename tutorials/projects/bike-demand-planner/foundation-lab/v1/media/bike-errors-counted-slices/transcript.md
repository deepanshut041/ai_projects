# Keep the denominator when you split the hours

## Ask which hours make the same score

The overall score combines quiet hours and busy commutes. To inspect that mixture, define exactly which rows belong to a slice. Rush clock hours here are seven, eight and nine in the morning, and sixteen, seventeen and eighteen in the afternoon. This window exists on every day. Working day rush adds the requirement that the working day flag is one. There are one thousand and ninety two rush clock observations, but only seven hundred and fifty working day rush observations. Mixing those definitions would attach the right number to the wrong population.

## Rebuild the overall MAE using the counts

The seven hundred and fifty working day rush observations have mean absolute error one hundred and seventy eight point one nine. All other three thousand, six hundred and seven observations average seventy three point five six. Multiply each mean by its observation count, add those error totals, and divide by four thousand, three hundred and fifty seven. This recovers the overall ninety one point five seven score. Simply averaging the two means would give the small rush population equal weight to the much larger remainder. The count tells us which objective the aggregate actually measures.

## Preserve error totals within every dimension

The implementation selects a slice by row indices and passes the same indices to predictions. It retains count, absolute error, squared error, residual, deficit and excess totals, rather than only rounded means. Each dimension must reconstruct the original population and those error totals. The independent audit verifies six hundred and eighty six slice summaries across seven model policy pairs and both partitions. Hour, month and day type each describe the same underlying observations. Adding counts across these overlapping dimensions would count hours repeatedly. Check each partition on its own.

## Keep time coverage beside the month score

January contains seven hundred and forty recorded validation hours and has mean absolute error thirty nine point six one. June contains seven hundred and twenty and has error one hundred and forty three. Mean signed error also becomes much more negative by June. These differences are hidden by the single aggregate. July through December contain zero validation observations, because those dates belong outside the retained window. We do not borrow reserved test outcomes to fill the table. A complete category domain can contain empty positions while the actual evidence remains limited to its observed dates.

## An empty category has no measured accuracy

Season names need the same coverage discipline. Winter here means January and February. Spring means March through May, while summer means only June. Autumn has no observations at all. Its metric is undefined and stored as null. If we displayed zero instead, the figure would misleadingly suggest perfectly predicted autumn hours. The chart explicitly labels no score. For root mean squared error, another denominator issue appears: first combine squared error totals, divide by count, then take the square root. Averaging seasonal root mean squared errors generally does not reproduce the overall value.

## Outcome bands explain failures after the fact

We also group errors by the eventual observed count. Hours below fifty rentals average error near twelve, while hours at least two hundred average about one hundred and sixty eight. That is useful retrospective evidence that the high count part of the population is difficult. Now consider using that band to choose a model when issuing a forecast. We cannot know the band without knowing the outcome. Turning this diagnostic into a routing input would leak the target into prediction. The slice remains an explanation of saved results, alongside its actual count and explicit retrospective scope.
