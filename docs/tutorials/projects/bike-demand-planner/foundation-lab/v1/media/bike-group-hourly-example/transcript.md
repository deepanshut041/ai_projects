# Choose a training mean for a real hour

## Start with the forecast we can issue

Start with Monday, the ninth of January, twenty twelve. At seven in the morning, we need a forecast for the hour beginning at eight. The target hour is eight, and the working day flag is one. Those calendar facts are already known. They form the pair eight, one. The recorded outcome was four hundred and twenty rentals. We can inspect that answer now because the hour is historical. At the original issue time, it was still in the future. Our predictor must select its answer without that outcome. The outcome becomes useful later, when we score the saved forecast.

## Recall the global reference

The existing constant predictor learned one number from the training year. Eight thousand, six hundred and forty five recorded hours contain one million, two hundred and forty three thousand, one hundred and three rentals. Dividing the total by the count gives about one hundred and forty three point seven nine rentals per hour. Every requested hour receives that same value. This makes a useful reference, but it cannot distinguish a quiet night from the morning commute. Missing source hours remain missing. We do not insert zero targets or change the denominator to a complete calendar grid.

## Average matching training hours

Now select the training hours whose key is eight, one. There are two hundred and forty eight of them. Their observed training targets total eighty six thousand, six hundred and seventy seven rentals. Divide that total by two hundred and forty eight. The group mean is about three hundred and forty nine point five zero rentals per hour. This is the number saved for a working day at eight. Calling predict with the pair eight, one looks up that saved value. We have changed the fitted function from one global level to calendar dependent levels. No validation answer participated in either fit.

## Score the quiet and busy hours

The working day group at two in the morning has a different training mean. Its total is one thousand, seven hundred and ninety six across two hundred and thirty nine hours. That gives seven point five one rentals per hour. The real later outcome is five, so the residual is positive two point five one. At eight, subtract four hundred and twenty from three hundred and forty nine point five zero. The residual is negative seventy point five zero, compared with negative two hundred and seventy six point two one for the global reference. Both examples become closer. We still need the complete validation comparison before making a broader improvement claim.

## Change the working-day status

Consider a different valid calendar request, eight in the morning on a nonworking day. The key becomes eight, zero. Its training group has one hundred and fifteen hours and eight thousand, seven hundred and ninety rentals. The saved mean is about seventy six point four three. The hour stays eight, but the selected parameter changes. This pairing can represent different daily profiles for working and nonworking days. It does not fit a separate Monday effect. Weekends and source holidays share the nonworking flag. That pooling is an assumption, and days sharing a key can still have very different outcomes.
