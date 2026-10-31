# Check the fitted function when coefficients are not unique

## The design has fewer independent directions

Our fitted design has forty six columns including the intercept, but its measured rank is forty two. Several columns repeat relationships already expressible by others. In raw coordinates, every row activates exactly one hour, weekday and month. Each complete indicator group therefore sums to the intercept column. In these training records, working day also equals the Monday through Friday indicators minus holiday. The independent audit checks that extra relation on this export. Standardization changes coordinates without removing those dependencies. The training data can determine a fitted function on supported inputs without assigning a unique value to every coefficient.

## Different coefficients can predict the same value

Consider a simple equivalent parameterization. Add ten to every raw hour coefficient and subtract ten from the intercept. Every valid hour vector has exactly one active hour indicator. The added ten and subtracted ten cancel. The forecast stays unchanged. That gives two different coefficient arrays for the same predictions. Requiring two solvers to print identical arrays would reject an equivalent representation. We compare their predictions and objective values instead. This argument also cautions against interpreting a single coefficient as a unique causal effect. Its numerical value depends on the chosen representation and the other columns.

## Use a stable independent training solve

The reference first uses the same saved training scaler. It appends ones and solves the training least squares problem with a stable numerical routine. We do not invert a normal matrix that may be singular. The routine returns parameters, rank and singular values. Its residual summary can be empty for a dependent design, so we explicitly calculate prediction errors instead of calling that empty array zero loss. The iterative and reference predictions differ by less than one millionth of a rental on observed training and validation inputs. Their training objectives agree to the displayed precision. That supports numerical correctness, while forecast usefulness remains a separate measurement.

## Convert the same predictor to raw coordinates

To inspect one hour, convert the standardized model back to raw coordinates. Divide each weight by its saved feature scale. Then subtract the mean weighted contribution from the standardized intercept to obtain the raw intercept. This conversion preserves the same function. In raw coordinates, inactive indicators are zero again, so Monday at eight can be read as an intercept plus four adjustments. The implementation checks equality between both coordinate systems. A constant training column uses scale one, which protects arithmetic but supplies no learned evidence for unseen variation. This is a readable decomposition of the saved representation, not a unique account of what causes demand.

## Read the saved Monday prediction

The saved raw intercept is about one hundred and forty three point five seven. The hour eight coefficient adds about one hundred and nineteen point five six. Monday adds three point seven three, January subtracts ninety two point seven nine, and working day adds about zero point seven one. Their sum is one hundred and seventy four point seven eight rentals per hour. This equals the standardized full vector calculation. If we shift every hour coefficient and compensate in the intercept, that total stays unchanged even though this displayed decomposition changes. Check the fitted function and its supported inputs before drawing conclusions from individual dependent coefficients.
