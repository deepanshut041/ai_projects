# Separate a valid calendar change from an unsupported request

## A valid input change can leave the output unchanged

Begin with our January Monday request at eight, supported by two hundred and forty eight training hours for its key. Change to Monday, June fourth, still at eight and still a working day. The forecast remains three hundred and forty nine point five zero. Both requests are valid. The selected predictor simply has no month term, so they share the same stored parameter. An unchanged prediction does not establish that January and June should have equal demand. It tells us what this fitted function can distinguish, which is a separate question from whether the input is well formed.

## Change a consistent calendar request

A valid Sunday request at eight selects the nonworking pair and returns seventy six point four three, supported by one hundred and fifteen training observations. The date, weekday and working day flag change together consistently. The Monday quiet hour uses another key and returns seven point five one. These comparisons expose the actual lookup behavior without fitting or reading an outcome. They are not causal estimates of changing one flag while holding an impossible calendar fixed. We retain four accepted cases with their support and fallback status so the model response can be traced concretely.

## A trusted flag can still pool unlike behavior

A large miss found afterward makes the pooling assumption tangible. On Monday, April sixteenth at eight, the source records a holiday and a nonworking flag. The model therefore uses the nonworking mean of seventy six point four three, while the observed count is five hundred and ninety seven. A holiday can be structurally consistent without behaving like a typical Sunday. We do not change the recorded flag afterward to obtain a better answer. This was a posthoc error example, not a predeclared extra request probe. The validator checks structure and consistency, while calendar authority must come from the trusted source.

## Reject a request whose time contract disagrees

The validator accepts exactly the schema version, target timestamp, issue timestamp and named calendar inputs. It checks exact hourly naive timestamps and requires the issue to be one hour before the target interval. It also checks that encoded hour, month and weekday agree with that timestamp. Named dictionary order is harmless because encoding follows the declared schema. Reordering the saved model feature names is rejected. Equal vector lengths cannot identify misplaced feature meanings. The fourteen rejected probes include missing and nonfinite fields, unknown categories and schema, extra outcomes, inconsistent flags, wrong lead and unsupported timestamp forms.

## Keep the supported replay domain explicit

The candidate currently supports historical validation replay from January first at one in the morning through June thirtieth at twenty three. The lower boundary follows the frozen one hour lead convention. Reserved test dates and a future twenty thirteen request are rejected even if their calendar values could be encoded. The dataset leaves timezone and daylight saving interpretation unspecified, so this review preserves naive timestamps rather than inventing an offset. This narrow domain records the evidence we actually have. A later application needs its own explicit serving and date policy; that adapter is not implemented by these sensitivity probes.

## Separate missing observation, missing group and invalid input

Three cases need different behavior. A valid hour with no source observation may receive an estimate, but no zero target is created for scoring. A valid hour and working day pair absent from training uses the saved training only global mean, one hundred and forty three point seven nine. All forty eight real pairs are present, so a small fixture checks this branch. A missing field or unknown schema instead causes rejection before lookup. Fallback is not permission to repair malformed input. Keeping those distinctions prevents coverage gaps and unsupported requests from being hidden behind an apparently successful numeric answer.
