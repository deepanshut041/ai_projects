# Reference results and how to read them

The values below come from the completed, disclosed historical project. They are not new fits performed for the tutorial. All comparison scores share their stated membership and output policy.

| Reference | Validation MAE | Later final MAE | Interpretation |
| --- | ---: | ---: | --- |
| Training mean | 154.2452 | See the complete final table in lesson 10 | One 2011 average for every request |
| Hour/working-day groups, raw | 91.5729 | 125.4167 | Original validation-selected candidate |
| Additive linear, clipped | 109.1436 | 121.0569 | Raw additive validation MAE is 116.7312 |
| Hour/working-day interactions, clipped | 100.5164 | 111.3046 | More flexible hourly curves |
| Representative forest, raw | 99.8324 | 103.6474 | Original representative seed; no later reselection |

Validation has 4357 observed hours and the final period has 4375. Training has 8645. Two boundary observations are embargoed; missing source slots are not zero-count targets. Seed-averaged study scores and individual representative-artifact scores answer different questions; keep them separate.

Follow [the final evaluation notes](lessons/10-final-evaluation-and-application.md) for the original seven-model table, signed error, monthly counts, rush-hour costs and the saved forecast-before-target sequence. Follow [setup](project/setup.md) for commands. Earlier test counts quoted in a lesson describe that build step; the completed source count is 128.
