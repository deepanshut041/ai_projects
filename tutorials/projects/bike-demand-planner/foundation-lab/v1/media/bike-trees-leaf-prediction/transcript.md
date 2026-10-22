# Follow one hourly request to a leaf mean

## Begin with four recorded training hours

Begin with four real observations from January fourth and fifth, twenty eleven. At two in the morning, the rental counts were one and two. At eight, they were one hundred and seventy nine and one hundred and ninety five. Calendar inputs can be known in advance, while these outcomes become available after their hours. For this small teaching probe, keep only the hour eight indicator. The later benchmark retains all forty five calendar columns. Our question is how a tree turns this declared input into a prediction.

## A constant cannot separate the two hours

First ignore the hour and fit a constant. The four counts sum to three hundred and seventy seven. Dividing by four gives ninety four point two five rentals per hour. This is the squared error minimizing constant for the whole tiny sample. It is much too high for the two quiet observations and much too low for the busy ones. The issue is not a poor learning rate. This constant family has no way to make the requested hour change its answer.

## One binary question creates two leaves

A one split tree can ask whether the hour eight indicator is at most zero point five. A zero takes the left branch. Its two supporting counts are one and two, so its mean is one point five. A one takes the right branch, whose counts are one hundred and seventy nine and one hundred and ninety five. Their mean is one hundred and eighty seven. Asking for eight now returns that right leaf value. The tree stores the question and the two values during fitting, then reuses them during prediction.

## Keep the probe calculation visible

The source selects the hour eight position from the four real encoded rows and keeps targets separate. It asks the library for a depth one regression tree with a fixed seed. Fitting constructs the test and mean leaves; prediction receives only zero or one. The library searches useful threshold splits rather than taking gradient steps on an existing linear weight vector. We use its implementation so the learning focuses on the model contract and arithmetic. The project also saves inspectable tests, support counts and values for the full models.

## Check the training reduction explicitly

The constant has total squared error thirty four thousand, five hundred and thirty eight point seven five on these four rows. Inside the busy leaf, each outcome is eight rentals away from its fitted mean, contributing sixty four each. The two quiet errors have magnitude one half, contributing one quarter each. The new total is one hundred and twenty eight point five. A mean leaf does not make its observations identical; it minimizes their squared deviations. This training improvement says nothing yet about another date.

## Ask for 09:00 using this restricted probe

Change the requested hour from eight to nine. In this probe, the only supplied indicator becomes zero. The tree therefore follows the same branch as the two oclock observations and returns one point five. That is not a claim about real nine oclock demand. It is what this deliberately limited fitted function can distinguish. The full benchmark uses all forty five calendar inputs and many more observations. It still cannot distinguish requests whose available inputs are identical. Keep this limitation in mind before interpreting deeper trees as automatic understanding of the future.
