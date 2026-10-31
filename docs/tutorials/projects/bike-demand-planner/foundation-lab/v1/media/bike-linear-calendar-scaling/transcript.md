# Follow Monday at eight into the linear model

## Start with a real hourly request

What does our linear model receive when it predicts an hour? Take Monday, the third of January, twenty eleven, at eight in the morning. This is training record fifty four. Under the retained convention, the request is issued at seven. It predicts observed rentals during eight to nine. The recorded outcome is one hundred and fifty four rentals, available after that interval. Calendar information is available in advance. The target and observed target hour weather are excluded from prediction inputs. We are following a real training observation, with the same information boundary used by the earlier baselines.

## Keep the same ordered calendar vector

The raw vector has forty five columns in a fixed order. Hour eight, Monday, January and the working day flag are active. Every other entry is zero, including the holiday flag. Record identity is stored separately. A raw coordinate linear forecast adds an intercept and the coefficients for these active inputs. Each adjustment is shared by many observations. Expanding the hour into separate indicators lets eight and two receive different effects, without assuming a constant increase for every hour number. Multiplying columns in the wrong order can produce a number with valid dimensions and the wrong meaning. The saved feature order is part of the predictor.

## Fit the numerical coordinates on training

The training loop uses standardized coordinates. For each column, subtract its training mean and divide by its population standard deviation. The hour eight indicator has a training mean near zero point zero four two, and a scale near zero point two zero one. For our active indicator, one minus that mean, divided by the scale, is about four point seven seven six six. Inactive columns become negative after centering. We must use the full standardized vector, rather than only its four originally active entries. The transformation changes numerical coordinates for optimization. It adds no information and leaves the target in rentals per hour.

## Save the scaler and reuse it

The source computes one mean and one population deviation per feature column. Axis zero reduces over training rows. The denominator is the full training count. A constant column receives scale one, protecting against division by zero. It then maps to zero on training. The model saves these values beside its weights. Prediction transforms new raw inputs with the saved statistics, then applies the dot product and intercept. The real training columns are not constant, but a small fixture checks that policy. Standardization does not remove dependent indicator columns. It makes the gradient coordinates more comparable while preserving the representable linear function.

## Change the request batch, not the model

Suppose a later request batch contains mostly eight oclock hours. Should we recompute the feature mean from that batch? We keep the saved training mean. The weights were fitted in those coordinates. A new batch mean would change the meaning of the same coefficient and could make one forecast depend on the other requests submitted beside it. Likewise, assigning scale one to a constant training feature does not teach its effect for a value never observed. That remains limited evidence. The raw exports stay unchanged. The scaler is learned state, fitted only on training and reused consistently for every later prediction.
