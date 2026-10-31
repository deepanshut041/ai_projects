# Shrink weights without pulling the intercept toward zero

## Add an explicit preference for smaller weights

A loss defines how misses contribute. L2 adds a preference among parameter vectors by charging for large weights. We use mean half squared error plus lambda over two times the squared weight norm. The intercept is excluded. There is no extra division of the penalty by the number of observations. Duplicating every row therefore leaves this objective unchanged. Its data mean and its penalty stay the same. These choices matter because a different normalization or a penalized intercept would define a different model.

## Work the penalty through one small update

Take illustrative weights two and minus three with lambda zero point two. Their squared norm is thirteen, so the penalty is one point three. Its gradient is lambda times each weight: zero point four and minus zero point six. If the data residuals are zero, a rate zero point one update changes the weights to one point nine six and minus two point nine four. An intercept of one hundred stays one hundred because it has no penalty contribution. This isolates the shrinkage calculation; it is not another fitted Bike result.

## Keep the data and penalty contributions visible

The source adds a separate scalar penalty to the mean data loss. The weight gradient receives lambda times weights. The intercept still receives only the mean slope. That is where the convention becomes executable. Independent finite differences check these gradients, and a duplicated data fixture checks the normalization. We define the penalty in standardized coordinates. The same numerical lambda on raw coefficients would mean a different preference. Scaling and regularization are therefore isolated in different study families, instead of treating a changed coordinate penalty as the same experiment.

## Solve the declared squared-error objective stably

To isolate regularization, we solve an augmented training least squares problem rather than adding another optimizer rate comparison. The appended weight rows are multiplied by the square root of training count times lambda. Their intercept column is zero. Dividing the augmented squared sum by twice the training count produces exactly our declared mean loss plus penalty. The displayed theta contains the intercept; the penalized weight vector excludes it. A stable numerical solve handles dependent columns without inverting a singular normal matrix. Independent stationarity checks verify that the resulting objective gradient is tiny.

## Shrinkage changes both the fit and later error

Increasing the declared penalty shrinks the standardized weight norm from about one hundred and five point five four to nine point nine three across this grid. Lambda zero point one slightly improves raw validation absolute error. Stronger values worsen it. After nonnegative clipping, no penalty scores better than zero point one: about one hundred and nine point one four versus one hundred and ten point four two. The output policy therefore belongs beside every result. A numerical loss including a penalty must also be separated from its data error when reading a table.

## Let the weights approach zero

When centered weights are pushed toward zero, the unpenalized intercept keeps the overall training level. The model approaches a constant forecast. At lambda ten, the quiet hour is predicted near one hundred and twenty four against target five. The busy hour is near one hundred and forty seven against four hundred and twenty. These outputs are positive but have lost useful hourly variation. Penalizing the intercept too would additionally pull the overall level toward zero, a different preference. Regularization trades fit for parameter control; this evidence does not make stronger shrinkage universally better.
