# Follow the same hourly error through three losses

## Start with the same recorded training hour

Return to Monday, the third of January, twenty eleven, at eight. The recorded target is one hundred and fifty four rentals. Zero weights and the completed training window mean give a prediction of about one hundred and forty three point seven nine. This is a training calculation, rather than a claim that those fitted statistics existed on January third. Prediction minus target gives residual minus ten point two one. The forecast is too low. We now keep that same miss and change how its loss values it.

## One miss produces three different pulls

Half squared error has slope equal to the residual, about minus ten point two one. Absolute error has slope minus one here. Huber with threshold ten uses a quadratic region near zero and a linear region outside it. The threshold is measured in rentals per hour. This residual is just outside, so its loss is ten times the absolute residual minus five, about fifty two point zero six. Its slope is capped at minus ten. The signs agree because the prediction is low; the magnitudes change how strongly this row influences fitting.

## Turn loss slopes into parameter gradients

The implementation returns one loss and one prediction slope for each row. It averages the loss values. Transposing the feature matrix and multiplying by slopes gives the weight contributions, divided by the row count. The intercept uses the slope mean. For our one row, the hour eight standardized value is about four point seven eight, so it multiplies each chosen slope. A full batch combines all eight thousand, six hundred and forty five rows, and their contributions can oppose one another. At an exact match, our MAE branch explicitly chooses subgradient zero.

## Change the target as a counterfactual

As an illustrative changed case, replace the target with four hundred and fifty four while keeping the same forecast. This is not a new recorded observation. The residual becomes about minus three hundred and ten point two one. Squared error now pulls much harder, while absolute error still uses minus one and Huber ten still uses minus ten. That is the mechanism behind lower sensitivity to extreme residuals. It does not prove that a busy commute should be ignored. A large real rental count may be exactly the behavior we need to predict.

## Tune within each loss before validation

Each objective receives the same three rates and two hundred training only pilot updates. The calendar representation, scaler, initialization and Adam convention stay fixed. We choose a rate by the final training objective within that loss. Every objective selects one here. The main fits reset state and run two thousand updates. We never compare the numerical value of MAE loss directly with Huber loss to choose a winner; their scales differ. Fixed step Adam can oscillate, so the final update budget is explicit and no exact nonsmooth optimum is claimed.

## Read the later errors and the individual hour

Squared error gives later mean absolute error about one hundred and sixteen point seven three. MAE fitting gives one hundred and seventeen point five six; the Huber variants are also slightly worse here. MAE did lower training absolute error. That does not guarantee lower error on another period. At the busy Monday target of four hundred and twenty, MAE and Huber ten forecast around two hundred and eighteen or nineteen, closer than squared error near one hundred and seventy five. Other observations offset that improvement. Keep the common validation metrics and raw failures visible before preferring a loss.
