# Build the 45 calendar inputs

Suppose a linear model receives the raw hour number eight. Its hourly contribution is one weight times eight. Moving from eight to nine always adds that same weight.

One straight slope cannot express a morning rise, a midday dip and an evening rise. There is also a clock boundary: twenty three and zero are adjacent in time, yet far apart numerically. We need to choose a representation based on what relationships the model should be able to learn.

The first representation allocates twenty four hour indicators, in hour order. At eight o'clock, the hour eight indicator is one. The other twenty three are zero.

A future linear model can assign a separate coefficient to each hour. This adds flexibility without inventing an ordinal distance between the category codes. Encoding creates a representation.

It does not fit any of those coefficients or guarantee that the resulting predictor will be accurate. We repeat that idea for seven weekday categories and twelve month categories. Then we append the holiday and working day flags, already zero or one.

The total is twenty four plus seven plus twelve plus two, which is forty five columns. The order is hour, weekday, month, holiday, working day. There is one shared schema for all partitions.

We save its column names so a later weight vector always refers to the same meanings. The hour columns occupy positions zero through twenty three. Hour eight activates position eight.

Weekdays begin at twenty four, so Monday, code one, activates position twenty five. Months begin at thirty one, so January activates position thirty one. Holiday at position forty three stays zero.

Working day at position forty four is one. The result has four ones and forty one zeros. The recorded target of one hundred fifty four stays outside these forty five values.

Read the encoder's inner expression as a membership question. Does this row's field equal this category? A match becomes floating point one, and any other category becomes zero.

The nested loop walks the category groups in the declared order. Afterward, the two flags are appended in their declared order. The code is short because the important decisions are explicit in the schema: permitted fields, category domains and column order.

In a later additive linear model, the bias contributes to every prediction. For this row, the active hour eight, Monday and January coefficients also contribute. The working day coefficient contributes because that flag is one.

The holiday coefficient is multiplied by zero. These terms describe the calculation that a fitted model would make. There are no fitted feature weights in this data preparation step.

One hot encoding permits separate category effects; this first additive form does not create hour by weekday interactions. Now change only the encoder's hour input from eight to nine. Hour eight switches off and hour nine switches on.

Monday, January and both flags stay unchanged. The feature count remains forty five. With fitted coefficients, the prediction difference would be the hour nine weight minus the hour eight weight.

This is a controlled encoding probe. It does not claim that the next real observation has the same target, or that demand must rise when the clock advances. One hot encoding treats hour categories separately.

It gives no built in preference for neighbouring hours to behave alike. A sine and cosine pair can represent position around the daily clock with two columns. Twenty three and zero then sit close together on a circle.

For a linear model, a single pair also constrains the daily contribution to one smooth sinusoidal shape. We keep one hot encoding as an inspectable first choice. A different encoding needs a controlled comparison on the same validation hours.

Let us follow the changed hour through exact coordinates. The Monday eight o’clock input activates positions eight, twenty five, thirty one and forty four. At nine o’clock, position eight turns off and position nine turns on.

The other active positions remain unchanged. Both vectors still have forty five values and four ones. Two coordinates changed, even though only one source field changed.

This is the concrete effect of replacing one categorical choice. To connect this representation to a linear prediction, choose illustrative coefficients. The intercept is ten, the hour eight coefficient twenty, and the shared Monday, January and working day contributions are three, two and five.

That sums to forty. Give hour nine coefficient twelve and the changed request sums to thirty two. The eight rental difference comes entirely from selecting a different hour coefficient.

These are hand chosen explanatory weights. They do not claim a trained prediction or accuracy for source record fifty four. Now suppose somebody exchanges the meanings of columns eight and nine but keeps the array length.

Would checking forty five columns and four ones detect the problem? Pause before answering. Both checks can still pass.

A weight vector would now refer to different hours, changing the calculation. The fixed feature names and saved schema order are therefore part of the model contract. Shape is necessary, but it is not the complete meaning of an input.
