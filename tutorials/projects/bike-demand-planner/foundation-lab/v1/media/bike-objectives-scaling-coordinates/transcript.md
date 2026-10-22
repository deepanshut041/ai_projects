# Change coordinates without adding a new pattern

## Represent the same hour in two coordinate systems

The Monday at eight vector has four active raw indicators. Training standardization maps the hour eight entry to about four point seven eight and also centers the inactive entries. The target remains in rentals per hour. No weather, interaction or new observation has been added. The fitted weights now operate in different numerical coordinates. We save the training statistics and reuse them at prediction time rather than recomputing them for each request batch. The question is how this changes optimization under a declared allowance, while keeping the input information fixed.

## Choose a learned scaler or the identity map

The implementation fits means and population deviations on training features. The raw branch instead stores zero means and unit scales. This excerpt calls that stored raw scaler identity. Both paths then use the same transform and prediction interface. Constant training columns receive unit scale in the standardized branch, preventing division by zero. Neither branch fits its transformation on validation. A later request batch cannot change the meaning of the saved coefficients. That fit and predict separation makes coordinate changes reproducible instead of silently refitting the model when requests arrive.

## Fix the update rule and the work

To compare conditioning, both paths use full batch SGD at rate zero point one for two hundred updates. They share half squared error, no penalty and the same initial prediction function. We do not give one path a new optimizer or extra passes. The parameter arrays are in different coordinates, so the same numerical step has different effects on the represented function. Raw and standardized gradient norms also have coordinate dependent units. A lower norm alone is not an invariant distance between predictors; training loss and an independent solve help interpret the trajectory.

## Read the fixed-work result

At the fixed allowance, raw indicators leave training half squared error around four thousand, one hundred and fourteen point five. Standardized coordinates reach about three thousand, one hundred and sixty seven point five three. Later absolute error is also lower for this standardized fit. Its final gradient is very small, but the configured stopping rule was still the update budget. We do not relabel the run as an early convergence experiment afterward. This is evidence of faster progress under these settings, not proof that raw indicators represent a permanently weaker family.

## Solve both coordinate systems independently

A stable training least squares solve in each coordinate system provides the missing comparison. On every observed training and validation input, their forecasts agree within a few trillionths of a rental. The raw model can represent the same fitted function on these inputs; its two hundred update result is unfinished. Dependent columns mean individual coefficients can differ. Coordinate dependent minimum norm solutions can also disagree on unsupported inputs. We therefore state the checked prediction agreement precisely. The result separates numerical conditioning from the additional feature capacity studied in the other families.

## Inspect an hour that favors the unfinished fit

The quiet Monday hour makes the distinction tangible. The raw two hundred step fit predicts about four point three nine against target five. The nearly optimized standardized function predicts minus seventy point six eight. The unfinished function is much closer on that observation, despite worse overall training loss and later absolute error. Optimizing the shared objective moves many rows together; it does not promise every individual row improves. Scaling also has not enforced nonnegative counts. Keep the single hour, the optimized training quantity and the later aggregate metric separate when explaining why the numbers changed.
