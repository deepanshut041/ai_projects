# The hourly prediction contract

Imagine Monday, the third of January, twenty eleven. At seven in the morning, we issue a bike rental forecast. The target is the hour from eight until nine.

The lead time is one hour before that interval starts. Our running example is source record fifty four. Its later recorded outcome is one hundred fifty four rentals.

That number helps us check a prediction after the interval finishes. The question throughout this tutorial is what information the original forecast could have used. The target hour is eight, the weekday code is one, and the month code is one.

In this source, Sunday is zero, so weekday one means Monday. Holiday is zero and working day is one. These calendar facts describe the upcoming interval.

The observed total is stored as cnt, with value one hundred fifty four. Calendar inputs describe the question. The later count supplies its answer.

Being stored beside each other does not give them the same role. We keep three roles separate. The record identifier and timestamps let us find and order an observation.

The permitted calendar fields become predictive inputs. The rental total becomes the separate training or evaluation target. Metadata stays attached to the row for traceability, outside the model feature vector.

A later loader selects named feature columns. Passing every numeric column into a model would lose this distinction and could introduce information from the answer. Consider a different observation, the first row, at midnight on the first of January.

It records three casual rentals and thirteen registered rentals. Their sum is sixteen, which matches the total. Checking this equality helps detect broken rows.

Supplying those same hour components as forecast inputs would reveal the target directly. We therefore validate the relationship and exclude both components from prediction. A perfect score obtained from the answer's components would not demonstrate a useful forecast.

Historical files also contain weather measured around the target hour. A measurement from eight to nine is available at a different time from a forecast issued at seven. The initial task uses calendar information available in advance.

Observed weather remains outside its inputs. A future weather experiment would need its own availability contract and a forecast source or an explicitly labelled historical comparison. A plausible looking feature can still answer the wrong timing question.

Let us deepen the contract with the next hourly request. The original forecast is issued at seven for the interval from eight until nine. Moving the requested interval to nine until ten moves a one hour lead issue to eight.

Monday, January, holiday and working day stay fixed. The requested hour changes. The old observed count of one hundred fifty four belongs to the old interval.

It cannot become the answer to the new request. Here is a small probe using the existing calendar encoder. The first mapping contains the five declared inputs.

The second replaces only the hour. Both calls produce forty five calendar values. Neither call needs the rental total.

Adding count to the mapping would trigger the exact input field check. This makes an information boundary executable. Metadata still retains issue and target time, so a later prediction can be matched to the request it actually answered.

Now ask a changed condition question. At eight o’clock, can we use the observed total for the interval from eight until nine? Pause and use the availability convention.

Its label has started, but the interval has not finished. The recorded total is therefore still a later outcome under our convention. The key distinction is the time information becomes available, rather than the position of its column or whether the hour label is familiar.
