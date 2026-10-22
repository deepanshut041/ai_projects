# Useful inputs, leakage and missing hours

The source is the University of California Irvine Bike Sharing dataset. Its creator and license are recorded with the source files. We retain its official archive and extract the hourly file and its readme.

Daily records remain separate. The acquired hourly file contains seventeen thousand three hundred seventy nine observations from twenty eleven and twenty twelve. We retain the acquisition details, attribution and license, together with hashes of the original bytes.

This gives later comparisons a specific dataset identity rather than only a download link. In the parser, the identifier becomes record I D. The date and hour become the target timestamp.

Validated source calendar fields stay together in a dictionary. This intermediate dictionary contains seven calendar fields. The observed total becomes the separate target.

The modelling encoder later selects five permitted fields from that dictionary. This split lets the parser validate source information without silently making every validated field a model input. Before returning a row, the parser checks all seventeen required fields and their allowed values.

Numeric values must be finite. The date must agree with year, month and weekday codes. Working day must agree with the weekend and holiday flags.

Component counts must add to the total. The file audit checks unique identifiers, unique hourly labels and strict chronological order. These checks establish a coherent set of observations.

They do not measure how well a demand model generalizes. One real source jump goes from January second at four in the morning to six. The five o'clock label has no row between them.

Two hours between recorded labels imply one unrecorded slot, calculated as two minus one. We leave that slot absent. Writing a new row with target zero would claim that no rentals occurred, which the file does not establish.

Complete cells in the rows we have do not imply a complete hourly time grid. The source labels span seven hundred thirty one calendar days. A naive grid has twenty four slots per day.

That gives seventeen thousand five hundred forty four slots. Subtracting seventeen thousand three hundred seventy nine observations leaves one hundred sixty five unrecorded slots, across seventy five gaps. Later we deliberately exclude two existing boundary rows for forecast timing.

Those are a different category. Missing rows are absent observations. Boundary exclusions are recorded observations whose evaluation issue times are too early.

The audit counts have a useful denominator. Seven hundred thirty one days times twenty four labels gives seventeen thousand five hundred forty four naive hourly slots. The difference from the recorded row count is one hundred sixty five, about zero point nine four percent.

That is a missing label fraction under this grid convention. It does not identify the missing outcomes, their cause, a time zone or a daylight saving policy. Those are additional claims that these counts do not establish.

Consider an explicitly illustrative window: ten, a missing row, and thirty. The two observed values average twenty. Inserting a zero makes three values whose average is about thirteen point three three.

The code shows that both the denominator and the dataset changed. We did not merely tidy a display. A later constant predictor or lag calculation can change because of that invented outcome.

Keep this separate from the actual recorded demand in the source. Suppose every cell in every available row passes validation. Does that prove that every hourly slot exists?

Pause and separate the two questions. A valid row can be followed by another valid row two hours later. Cell level consistency and time grid coverage are different checks.

Preserve the distinction between a recorded zero and no recorded observation. Choosing a justified missing data policy is later work, rather than a silent parser assumption.
