# Keep the forecast and recorded outcome separate in the application

## Let the user ask for an estimate before an outcome

The replay interface starts with the same Monday eight o’clock request. It first resolves consistent calendar metadata, then sends a target free request to prediction. The response contains the estimate, simple baseline, support and model identities. It contains no observed count or residual. A separate show outcome action retrieves a historical observation for comparison. The server holds that separate record for replay, while the predictor can run without it. Browser checks confirm the client makes no observation request before the explicit action. This gives the information boundary visible behavior instead of leaving it only in a comment.

## Inspect the completed historical comparison

This screenshot shows the completed comparison after the outcome action. The forecast card displays three hundred and forty nine point five rentals per hour. The global training mean is one hundred and forty three point eight, and the fitted group has two hundred and forty eight supporting hours. The separate outcome card displays four hundred and twenty with signed error negative seventy point five zero. The interface rounds for reading, while the API retains the full floating point values. Issue time, schema, used fields and model and bundle hashes remain inspectable so the estimate has an identifiable contract.

## Keep the two operations separate in the adapter

The adapter obtains a consistent request and sends it through the saved predictor. Prediction reuses fitted state, and no button click or server startup fits another model. The separate observation lookup validates the same historical date boundary and can return unavailable. Application data is checked against the predictor and dataset identities before serving. Preparation is an explicit earlier command, separate from loading and serving the saved state. The HTTP server binds to loopback and serves fixed packaged assets and routes. It does not turn a browser supplied string into a file path or training command.

## A real missing hour still has no target

Change to January second at three in the morning, a genuinely unrecorded validation hour. The recorded date metadata still supplies a valid calendar request. The model can therefore produce an estimate from its fitted group. The observation lookup returns unavailable with no target, and the interface says an actual count and error are absent. It does not treat the missing count as zero rentals or add a row to the benchmark denominator. An available estimate and unavailable observed evidence can coexist. Handling both explicitly keeps the replay faithful to the source rather than manufacturing a complete calendar grid.

## An unsupported request clears the old result

Now change to a future twenty thirteen date. That is outside the historical validation domain, even though its hour could be encoded. The backend rejects it, and the browser clears the old forecast instead of leaving three hundred and forty nine point five looking like the new answer. Reserved test dates, non hourly timestamps, unknown schema and a request containing an outcome also fail. These errors differ from a valid missing observation. The working application preserves the earlier information and support boundaries, while the explicit message helps the user understand why no forecast is being displayed.

## Verify the actual interface and the installed package

Actual browser checks exercise quiet Monday, Sunday and June requests, the observed comparison, missing evidence and rejected dates. The layout passes at four widths from fourteen hundred and forty to three hundred and twenty pixels without document overflow or JavaScript errors. An isolated installed wheel serves its packaged HTML, JavaScript and CSS while an audit hook denies all source project access. The user can run the application from saved assets rather than an active training notebook. That is implemented local behavior. It does not establish hosted recovery, operational usefulness or a final held out assessment.
