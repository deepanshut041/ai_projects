# Final narration scripts

The complete current narration follows; earlier draft and revised scene versions remain in the production archive.

# 1 · Defend the answer with its evidence

## question {hold=1.5 transition=fade}
[#a] The Lumen manual says a full charge takes three hours. [#b] Our question asks how long that full charge takes. [#c] The assistant answers three hours, but cites the power-button passage. [#d] Its answer text matches the fact we expected. [#e] The cited passage exists, yet it describes starting the lamp. [#f] We need to check the claim and its supporting evidence separately.

## reasoning {hold=1.5 transition=fade}
[#a] Correctness compares the claimed fact with the reviewed answer. [#b] Citation validity checks that the supplied document contains that identifier. [#c] Support asks whether the cited text establishes the material claim. [#d] A valid identifier can still refer to unrelated text. [#e] The acceptable answer must pass all three checks together. [#f] Compression can change a fact, a citation, or the willingness to guess.

## code-01-contract-py {hold=1.5 transition=fade min=32}
[#p0] The complete program starts with its three supplied passages. Each identifier names a distinct manual fact, so the charging and power-button evidence cannot be interchanged. [#p1] The gold question declares its expected answer and evidence set. Score normalizes fixture text and checks cited identifiers before constructing its support condition. [#p2] The function returns separate correctness, validity and support flags. The main fixture then prepares both the correct-source response and the deliberately unrelated citation. [#p3] Both calls share the same gold question and passages. Assertions require the good record to pass and the unrelated citation to fail support before printing the result.

## result {hold=1.5 transition=fade}
[#a] The complete reference scores two deliberately supplied response records. [#b] Both records contain the same three-hour answer. [#c] The first cites the charging passage and passes the fixture support check. [#d] The second cites the power-button passage and fails that check. [#e] This is checked application logic over authored data. [#f] It does not load a language model or establish general semantic entailment.

## transfer {hold=1.5 transition=fade}
[#a] Change the answer to three hours and a five-year warranty. [#b] The charging part still matches the manual. [#c] The warranty duration appears nowhere in the supplied passages. [#d] A complete-answer support review must therefore reject the extra claim. [#e] Our first exact fixture scorer needs a richer rubric for this case. [#f] Every material claim needs support, even when the expected short answer appears.

## bridge {hold=1.5 transition=fade}
[#a] Keep the exact question, supplied passages and raw response for each request. [#b] Record answer status, citations and finish reason beside the artifact identity. [#c] A refusal can be correct when the document lacks the requested fact. [#d] It becomes a false refusal when the evidence was actually supplied. [#e] These outcomes will guide the later quality and compression studies. [#f] Next we inspect which project commands run today.


---

# 2 · Trace the runnable project scaffold

## question {hold=1.5 transition=fade}
[#a] The Document Assistant already has runnable setup commands. [#b] The status action prints the proposed Gemma checkpoint and experiment defaults. [#c] That output establishes configuration access and command execution. [#d] It does not establish that any model weights can load. [#e] We will trace the existing source before enabling a real inference action. [#f] This keeps a successful setup check from becoming an invented model result.

## reasoning {hold=1.5 transition=fade}
[#a] The configuration loader reads a packaged TOML resource or an explicit file. [#b] It validates ownership, exact model revision and positive token budgets. [#c] The workflow module declares which model actions are still pending. [#d] The dispatcher uses those definitions to choose output and exit status. [#e] The thin entry point finally calls that dispatcher. [#f] We show each complete definition before the code that calls it.

## code-config-py {hold=1.5 transition=fade min=48}
[#p0] This is the complete existing configuration loader before its dispatcher calls it. It imports packaged-resource access and declares the owning project and positive budget fields. [#p1] Load-config reads either the explicitly supplied file or the packaged TOML defaults. Parsed values are then checked for the declared schema and project ownership. [#p2] The model and optional fallback sections are validated as mappings. Each repository identifier must be nonempty before its exact hexadecimal revision is checked. [#p3] Missing or malformed revision identity raises an error. The source compute type is restricted to the declared floating choices before experiment validation begins. [#p4] The experiment must contain a nonnegative integer seed and each required positive token budget. Exact integer checks prevent a Boolean being accepted as a numeric budget. [#p5] The final loop rejects invalid positive fields and returns the parsed configuration. No model framework, network client or weight acquisition is called by this loader.

## code-workflow-py {hold=1.5 transition=fade min=16}
[#p0] The workflow declares prepare-data, quantize, evaluate and answer as pending. Pending-result validates the requested action before constructing a structured status response. [#p1] The returned status explicitly says not implemented and states that no model or output was created. This helper is shown before the command layer uses it.

## code-commands-py {hold=1.5 transition=fade min=48}
[#p0] The dispatcher imports the already shown loader and pending-action helpers. Its main function begins by declaring the command-line parser and project description. [#p1] Parser choices expose the three setup actions and explicit future actions. The optional configuration argument selects an alternative public project configuration file. [#p2] After parsing, configuration is loaded. A future model action prints the pending record to standard error and returns exit two without producing an answer. [#p3] Show-config returns the parsed configuration. Other setup actions construct a boilerplate status record with pinned model defaults and model-execution-verified false. [#p4] Check adds interpreter, configuration and source-import observations. Successful JSON is printed to standard output and the command returns zero. [#p5] File and validation exceptions produce an error record and exit one. These output streams and codes are the real scaffold behavior the controller must interpret.

## code-document-assistant-py {hold=1.5 transition=fade min=8}
[#p0] The complete direct entry point now imports the previously shown main function and converts its return value into process exit status. It contains no hidden model work.

## code-02-scaffold-trace-py {hold=1.5 transition=fade min=40}
[#p0] The live inspector receives an explicit project root. It constructs the project interpreter and entry-point paths, then declares the four actions and expected exit codes. [#p1] Each subprocess runs from a different working directory and captures both streams. The code chooses standard output for successful setup and standard error for the pending action. [#p2] The parsed payload is recorded with its project identity and exit status. Assertions distinguish pending answering from setup reports that explicitly deny verified model execution. [#p3] The optional command-line argument activates the real project inspection. Without it, the portable course run prints only its limited expected contract. [#p4] The final closing branch completes that portable JSON response. A no-argument run is therefore not evidence that a live checkout or model was inspected.

## result {hold=1.5 transition=fade}
[#a] Status, setup check and configuration should complete with exit zero. [#b] Their structured output goes to standard output. [#c] A requested answer currently returns exit two with a pending message. [#d] That message goes to standard error and creates no model result. [#e] Configuration or file failures use a separate error status. [#f] The live trace runs these actual scripts from another working directory.

## transfer {hold=1.5 transition=fade}
[#a] Open Projects, Quantization Lab, then Document Assistant. [#b] Select status, setup check or configuration and press Run. [#c] Run starts the program; Checks runs tests. [#d] Successful status still leaves model loading and generation unproven. [#e] Imports must work outside the source folder. [#f] A pending action must never promise an answer.

## bridge {hold=1.5 transition=fade}
[#a] The scaffold gives us a controlled place to add real work. [#b] Data preparation will belong in the data package. [#c] Artifact recipes and evaluation retain separate responsibilities. [#d] The current model actions remain pending until those implementations exist. [#e] Keep these source files and their existing tests unchanged during video production. [#f] Next we define what the selected model and runtime must prove.


---

# 3 · Prove model and runtime compatibility

## question {hold=1.5 transition=fade}
[#a] Our proposed source is Gemma four E four B instruction tuned. [#b] The project stores an exact revision rather than relying on a moving branch. [#c] A runtime still has to interpret its architecture and special tokens correctly. [#d] A configuration parser succeeding is weaker evidence than generated output. [#e] We need a bounded probe before downloading or benchmarking a full experiment. [#f] The course treats actual compatibility as pending until that probe exists.

## reasoning {hold=1.5 transition=fade}
[#a] First prove artifact loading at the declared placement. [#b] Then check rendered prompt tokens and the intended chat template. [#c] Generation must honor end markers and its output allowance. [#d] A new process must reload the same verified artifact. [#e] With two thousand forty-eight prompt tokens and two hundred fifty-six output tokens, capacity must cover two thousand three hundred four. [#f] The publisher context limit alone does not establish a feasible local budget.

## code-03-compatibility-py {hold=1.5 transition=fade min=40}
[#p0] Require-ready first checks an exact hexadecimal revision and declares the five capabilities. These fields describe actual future probes, not a model load performed by this function. [#p1] The missing-capability list requires each probe to be true. A separate capacity check includes prompt and output tokens before any acceptance result is returned. [#p2] The main function constructs a deliberately synthetic passing manifest with its explicit prompt, output and context values. All fixture capability names are supplied. [#p3] The fixture first satisfies the gate, then removes template agreement. The next call must raise, and the exception message is retained as the expected rejection. [#p4] A missing exception would fail the test. The printed result identifies its fixture scope and missing capability, while the entry point only invokes this validator demonstration.

## result {hold=1.5 transition=fade}
[#a] The reference manifest first contains synthetic passing probe fields. [#b] Its acceptance means only that those fixture fields satisfy the gate. [#c] We then clear the template agreement field. [#d] The validator must reject the record and name that missing capability. [#e] In a real receipt, each true field points to an executed command and retained output. [#f] No model execution is inferred from a Boolean written into a file.

## transfer {hold=1.5 transition=fade}
[#a] Suppose the backend loads weights but uses a different chat generation prefix. [#b] The user-visible question can still look identical. [#c] The model nevertheless receives a different token sequence. [#d] Fix the template or declare a separately versioned input policy. [#e] Capture prompt bytes, token identifiers and special-token behavior before comparing weights. [#f] A load-success message cannot repair a prompt mismatch.

## bridge {hold=1.5 transition=fade}
[#a] The pinned model metadata includes hybrid attention and cache sharing. [#b] It also distinguishes local and global attention dimensions. [#c] Those facts require a more careful cache model later in the course. [#d] Source support in a runtime repository does not prove the selected device path works. [#e] A model fallback requires its own source identity and baseline. [#f] Next we budget weights and request state that are live together.


---

# 4 · Budget a complete request

## question {hold=1.5 transition=fade}
[#a] A compressed model file may fit while the actual request fails. [#b] Weights share memory with cache, temporary workspaces and other allocations. [#c] We also reserve headroom rather than planning to use every byte. [#d] The quantities must belong to the same host or device memory pool. [#e] Disk bytes and peak live memory answer different questions. [#f] We start with explicit illustrative numbers before measuring a real backend.

## reasoning {hold=1.5 transition=fade}
[#a] Assume eight billion stored scalar values as a hypothetical inventory. [#b] Two-byte storage needs sixteen billion payload bytes. [#c] Four-bit payload needs four billion bytes. [#d] A two-byte scale for every thirty-two values adds half a billion bytes. [#e] That makes four point five billion bytes, about four point one nine one gibibytes. [#f] The scale overhead means this toy representation is not an exact fourfold reduction.

## code-04-memory-budget-py {hold=1.5 transition=fade min=24}
[#p0] The helper counts packed payload with a ceiling and adds one scale record for every group. Main declares the hypothetical parameter inventory before computing floating storage. [#p1] The same helper computes four-bit storage. The complete budget then adds cache, workspace and reserve, and checks the illustrative capacity inequality. [#p2] The last field repeats feasibility with an eight-gibibyte cache allowance. The output distinguishes a fitting short-cache plan from the failing longer-cache plan.

## result {hold=1.5 transition=fade}
[#a] Give the toy weights a two-gibibyte cache allowance. [#b] Add one gibibyte of workspace and two gibibytes of reserve. [#c] Total planned usage including reserve is nine point one nine one gibibytes. [#d] It fits within the illustrative twelve-gibibyte capacity. [#e] Actual loading and conversion can have different temporary peaks. [#f] Measure each operation and distinguish allocated, reserved and resident memory.

## transfer {hold=1.5 transition=fade}
[#a] Now increase cache from two to eight gibibytes. [#b] The weight artifact and its scale metadata stay unchanged. [#c] Planned usage rises by six gibibytes to fifteen point one nine one. [#d] The same compressed model now exceeds the twelve-gibibyte budget. [#e] Reducing weight bytes did not remove the context-state constraint. [#f] Keep this changed variable separate from the later weight experiment.

## bridge {hold=1.5 transition=fade}
[#a] Record before-load, load peak and post-load steady memory. [#b] Then measure prefill, decode and retained state after the request. [#c] Do not add peaks that never occur at the same time. [#d] Do not omit a conversion buffer that overlaps both source and destination. [#e] A fresh process helps distinguish retained reservation from per-request growth. [#f] Next we give document text and evidence spans stable identities.


---

# 5 · Make document evidence reproducible

## question {hold=1.5 transition=fade}
[#a] The Lumen charging sentence must remain traceable after parsing and reruns. [#b] A changed line ending should not silently create a different logical manual. [#c] A changed charging duration must invalidate the old document snapshot. [#d] We therefore keep original bytes and define a canonical text transformation. [#e] The first supported scope is plain text or Markdown. [#f] A later PDF importer needs separate reading-order and extraction checks.

## reasoning {hold=1.5 transition=fade}
[#a] Normalize line endings and the declared Unicode representation. [#b] Hash the canonical UTF-eight text to identify the complete document. [#c] The first fixture passage occupies characters zero through twenty-seven. [#d] Its interval is written zero inclusive to twenty-eight exclusive. [#e] After the blank-line separator, the second occupies thirty through forty-five. [#f] Character offsets, byte offsets and model-token offsets are different coordinate systems.

## code-05-passages-py {hold=1.5 transition=fade min=32}
[#p0] Canonicalization normalizes declared line endings and Unicode form, then supplies one terminal newline. Passage construction calls that defined transformation before hashing text. [#p1] The digest identifies the complete canonical document. A forward cursor locates each nonempty paragraph after the previous occurrence and begins its provenance record. [#p2] Each record retains text and character offsets, then checks the source-span round trip. Main compares line-ending variants and a deliberately changed charging fact. [#p3] The printed offsets and stability flags summarize those exercised invariants. Every helper and transformation used by main has appeared before the calling fixture.

## result {hold=1.5 transition=fade}
[#a] The reference produces the same records for the two line-ending variants. [#b] It searches for the next paragraph after the previous occurrence. [#c] That cursor prevents repeated text from always pointing to the first copy. [#d] Changing three hours to four changes the document hash and passage identities. [#e] The shortened teaching identifier still retains a full hash in the record. [#f] A production key needs a complete digest or an explicit collision policy.

## transfer {hold=1.5 transition=fade}
[#a] Place the same warning paragraph twice in one manual. [#b] A search starting from zero can attach both records to the first warning. [#c] A forward cursor preserves the distinct source intervals. [#d] Identical wording therefore does not imply an identical occurrence. [#e] Content-only identifiers need a separate location policy when text repeats. [#f] The source-span round trip remains a required invariant.

## bridge {hold=1.5 transition=fade}
[#a] A large paragraph may still exceed the tokenizer budget. [#b] A deterministic chunker must preserve parent section and source spans. [#c] Overlap can repeat a fact without creating independent evidence. [#d] Avoid separating a measurement from its unit or a table value from its heading. [#e] Freeze the chunking policy before comparing compressed models. [#f] Next we separate calibration, development and final document roles.


---

# 6 · Freeze data roles and prompt budgets

## question {hold=1.5 transition=fade}
[#a] Stable hashes let us identify documents, but they do not prevent overlap. [#b] Two differently worded questions may use the same facts in one manual. [#c] Splitting questions alone can expose final evidence during calibration or tuning. [#d] We assign whole document groups to their roles first. [#e] Related versions and copied manuals also need grouping. [#f] Only then do we attach question labels and evaluate candidates.

## reasoning {hold=1.5 transition=fade}
[#a] Let overhead include instructions, question and template control tokens. [#b] The fixture overhead is three tokens and the first passage contributes four. [#c] Their total of seven fits the eight-token prompt cap. [#d] Adding the second two-token passage would make nine and is rejected. [#e] With four output tokens, total used capacity would be eleven. [#f] The real implementation counts the complete rendered model prompt, not whitespace words.

## code-06-splits-py {hold=1.5 transition=fade min=32}
[#p0] Split maps a seeded document identifier into a deterministic hash bucket. The thresholds assign a document role without consulting question labels or model scores. [#p1] Pack considers passages in fixed order. It retains a candidate only if the prompt cap and total prompt-plus-output capacity both hold. [#p2] Main verifies document membership and disjointness, then supplies a whitespace counter solely for the visible packing fixture. The actual model requires its own complete-prompt tokenizer. [#p3] The expected packed list retains only the first passage under these small bounds. Assertions and the final JSON report the fixture result without preparing a real dataset.

## result {hold=1.5 transition=fade}
[#a] The reference hashes a seed and document identifier into a fixed bucket. [#b] Buckets map to calibration, development and final membership. [#c] It checks that every document belongs to one role. [#d] A small dataset may still have unbalanced domains or answerability. [#e] Inspect and freeze grouped membership before choosing recipes. [#f] Do not search split seeds to obtain a more favorable score.

## transfer {hold=1.5 transition=fade}
[#a] Suppose packing drops the passage containing the charging time. [#b] A model refusal may now be correct for the supplied prompt. [#c] That does not establish a compression failure on an answerable complete document. [#d] Retain selected and dropped passage identifiers with their reasons. [#e] For a weight study, verify the labeled evidence was supplied to every candidate. [#f] Implicit backend truncation would hide the cause of the answer change.

## bridge {hold=1.5 transition=fade}
[#a] Retain source provenance, licenses, canonical hashes and role membership. [#b] Review answerability, accepted facts and alternative supporting spans. [#c] Calibration text generally does not need answer labels. [#d] Development chooses recipes while the final assessment remains untouched. [#e] Retrieval is a separately versioned later component. [#f] Next we establish the floating-point generation behavior to preserve.


---

# 7 · Preserve the floating-point reference

## question {hold=1.5 transition=fade}
[#a] The same visible question can reach a model through different token sequences. [#b] We freeze the tokenizer, chat template and complete packed prompt. [#c] The model then produces a score for every possible next token. [#d] Those logits are unnormalized scores rather than probabilities. [#e] We need a precise rule for selecting tokens and stopping generation. [#f] The first comparison uses a declared greedy policy and records its failures.

## reasoning {hold=1.5 transition=fade}
[#a] Take logits zero and the natural logarithm of three. [#b] Their exponentials are proportional to one and three. [#c] After dividing by four, the probabilities are one quarter and three quarters. [#d] Adding one thousand to both scores preserves that ratio mathematically. [#e] Subtracting the maximum before exponentiation avoids a large exponential overflow. [#f] Greedy selection can take the largest logit directly without calculating probabilities.

## code-07-reference-decode-py {hold=1.5 transition=fade min=32}
[#p0] Softmax subtracts the maximum before exponentiation and normalization. Decode then selects the largest supplied logit directly and checks whether that identifier is the end marker. [#p1] The decoder appends only content tokens and records why it stopped. Main compares ordinary and shifted score vectors before supplying the small invented token sequence. [#p2] Assertions require content identifiers zero and one followed by an end-marker reason. Another assertion exercises the length cap, and JSON prints probabilities and finish state. [#p3] The entry point only runs these supplied-logit checks. No tokenizer, weight artifact or trained language model is hidden behind the decoding helper.

## result {hold=1.5 transition=fade}
[#a] Our invented vocabulary has token zero for three and token one for hours. [#b] Token two is the end marker. [#c] The supplied logit sequence selects zero, one, then two. [#d] The content result contains only zero and one, with an end-marker finish reason. [#e] A one-step output cap instead produces a length-limited result. [#f] These supplied logits explain mechanics and are not output from a trained model.

## transfer {hold=1.5 transition=fade}
[#a] Suppose the leading logits are two point zero zero and one point nine nine. [#b] A differential shift of two hundredths can reverse their order. [#c] Greedy decoding then appends a different first token. [#d] Later distributions are conditioned on that changed history. [#e] Low average tensor error therefore does not guarantee identical answers. [#f] A seed also does not guarantee identical kernels or decisions across runtimes.

## bridge {hold=1.5 transition=fade}
[#a] Retain raw text, output token identifiers and parsed answer structure. [#b] Keep the artifact hash, runtime build, placement and prompt identity. [#c] A converted floating artifact needs a bounded agreement probe against its source. [#d] Declare its token or numerical acceptance contract before inspecting differences. [#e] The reference remains the quality anchor during compression. [#f] Next we derive how a finite integer grid approximates a floating value.


---

# 8 · Derive scale, zero point and clipping

## question {hold=1.5 transition=fade}
[#a] A three-bit code can identify only eight possible bit patterns. [#b] To approximate a floating weight, each pattern needs a reconstructed value. [#c] The scale sets the spacing between those values. [#d] The zero point identifies which integer reconstructs real zero. [#e] Encoding also needs a rounding rule and saturation limits. [#f] An integer cast alone does not specify this quantization operation.

## reasoning {hold=1.5 transition=fade}
[#a] Choose unsigned codes zero through seven for a desired range minus two to four. [#b] The step is the six-unit range divided by seven intervals. [#c] The rounded zero point becomes two. [#d] Code zero reconstructs minus twelve sevenths, while code seven reconstructs thirty sevenths. [#e] Real zero is exact, but the desired endpoints shift because the zero point is integer. [#f] Input one maps to code three and reconstructs six sevenths.

## code-08-scalar-quant-py {hold=1.5 transition=fade min=32}
[#p0] Affine validates a nonzero range containing zero, derives the scale and rounded zero point, then clips each rounded code into the chosen unsigned interval. [#p1] Reconstruction restores original units. The symmetric helper separately declares its signed range and handles an all-zero block before main begins the fixture. [#p2] Main checks the known codes, zero-block behavior and ties-to-even rounding. The output retains the scale, zero point, codes and restored values for direct inspection. [#p3] The remaining print and entry-point lines complete that same deterministic example. The visible quantizer rule is scoped to the teaching fixture, not a GGUF preset implementation.

## result {hold=1.5 transition=fade}
[#a] Input seven exceeds the represented range and saturates at the largest code. [#b] It reconstructs about four point two eight six, losing the original magnitude. [#c] For an unclipped nearest rounding, error is at most half the step. [#d] Clipping outside the range is not covered by that bound. [#e] The symmetric reference also handles an all-zero block without dividing by zero. [#f] A real format can use a different signed range or block representation.

## transfer {hold=1.5 transition=fade}
[#a] Use scale one half, zero point zero, and codes minus seven through seven. [#b] Input one point two five becomes code-unit value two point five. [#c] Ties to even chooses code two and reconstructs one point zero. [#d] Input ten would need code twenty, so it clips to seven and reconstructs three point five. [#e] The first error is minus one quarter; the second is minus six point five. [#f] Always identify division, rounding, saturation and reconstruction as separate steps.

## bridge {hold=1.5 transition=fade}
[#a] Weight-only four-bit storage can still multiply sixteen-bit activations. [#b] A kernel may unpack blocks and accumulate at another precision. [#c] Record weights, activations, cache and accumulation types separately. [#d] For quantized weights and activations, their two scales multiply the integer dot product. [#e] Our project begins with post-training compression rather than a newly trained model. [#f] Next we store the codes in real packed bytes and count their metadata.


---

# 9 · Pack low-bit weights and count overhead

## question {hold=1.5 transition=fade}
[#a] The scalar quantizer produced small integer codes. [#b] A container of full integer objects still has object and container overhead. [#c] We now define an actual four-bit byte layout in C plus plus. [#d] Two nibbles occupy the low and high halves of one byte. [#e] The decoder must know their order and the original number of values. [#f] This toy layout remains separate from real GGUF block formats.

## reasoning {hold=1.5 transition=fade}
[#a] Take the codes minus eight, minus one, zero, seven and three. [#b] Their low four-bit patterns are eight, F, zero, seven and three in hexadecimal. [#c] The first pair produces hexadecimal F eight, decimal two hundred forty-eight. [#d] The second pair produces hexadecimal seven zero, decimal one hundred twelve. [#e] The odd final code leaves padding and produces decimal three. [#f] Without the count, the unused high nibble would become an invented sixth zero.

## code-09-pack-int4-cpp {hold=1.5 transition=fade min=32}
[#p0] The C plus plus packer allocates enough bytes for odd counts. It rejects out-of-range signed codes before extracting their low four-bit patterns. [#p1] The code shifts each nibble into its declared half-byte. Unpack validates byte count and extracts the matching nibble from each stored position. [#p2] Patterns eight through fifteen are converted back to negative values. Main checks the five-value round trip and verifies that an out-of-range code is rejected. [#p3] The final output prints byte values as integers rather than characters. The expected three bytes are two hundred forty-eight, one hundred twelve and three.

## result {hold=1.5 transition=fade}
[#a] The packer validates that every value belongs to minus eight through seven. [#b] Unpacking extracts a nibble and subtracts sixteen for patterns eight through fifteen. [#c] The known five-value round trip must reproduce every signed code. [#d] A code of eight is rejected rather than silently becoming minus eight. [#e] The recovered integers still need their scale and zero point to reconstruct weights. [#f] A successful load alone would not detect a sign or nibble-order mistake.

## transfer {hold=1.5 transition=fade}
[#a] Use four payload bits and a two-byte scale for each group of thirty-two values. [#b] Those sixteen metadata bits add one half bit per weight. [#c] The effective count becomes four point five bits before other overhead. [#d] Adding another two-byte zero point raises it to five bits. [#e] A group of one hundred twenty-eight with only the scale needs four point one two five bits. [#f] Larger groups save metadata while forcing more weights to share a range.

## bridge {hold=1.5 transition=fade}
[#a] Real artifacts also store tensor shapes, offsets, alignment and type identities. [#b] Some tensors can stay at higher precision under a mixed preset. [#c] On-the-fly unpacking and large float expansion have different memory costs. [#d] Inspect actual artifact composition and the selected inference path. [#e] Do not rename our toy bytes as a model container. [#f] Next we trace reconstructed weight error into the output of a layer.


---

# 10 · Trace quantization error through a layer

## question {hold=1.5 transition=fade}
[#a] A compressed matrix is used inside a computation rather than judged alone. [#b] Rows of our weight matrix produce output channels. [#c] Columns multiply the corresponding input channels. [#d] The input is four in its first channel and one in its second. [#e] Errors in the first weight column therefore receive a larger multiplier. [#f] We will calculate both floating and reconstructed outputs with every number visible.

## reasoning {hold=1.5 transition=fade}
[#a] The first floating row is zero point two four and one point zero two. [#b] Its dot product is zero point two four times four plus one point zero two, giving one point nine eight. [#c] The second row gives minus two point five five. [#d] Reconstructed rows zero and one, then minus one and one half, produce one and minus three point five. [#e] Subtracting the floating outputs gives minus zero point nine eight and minus zero point nine five. [#f] The same difference equals the weight-error matrix multiplied by the unchanged input.

## code-10-layer-error-py {hold=1.5 transition=fade min=24}
[#p0] Matrix-vector multiplication validates input width before summing products. Main declares both weight matrices and the unchanged input, then starts building their error matrix. [#p1] The program computes both outputs, subtracts them and independently evaluates the error matrix times the input. It also checks the norm-based diagnostic bound. [#p2] The printed vectors expose floating output, reconstructed output and their signed difference. All matrix helpers are defined before the fixture calls them.

## result {hold=1.5 transition=fade}
[#a] The reference computes each matrix-vector result before subtracting outputs. [#b] It separately multiplies the error matrix by the input. [#c] Those two calculations must agree. [#d] A norm bound also limits output error by matrix-error norm times input norm. [#e] The bound explains sensitivity but does not predict a document answer score. [#f] The separate rational audit checks the visible decimal arithmetic exactly.

## transfer {hold=1.5 transition=fade}
[#a] Change the input to one in the first channel and four in the second. [#b] The first output error becomes minus zero point two four minus zero point zero eight. [#c] That is minus zero point three two. [#d] The second becomes minus zero point two four plus zero point zero four, or minus zero point two zero. [#e] The weight-error matrix and its mean squared error have not changed. [#f] Input-dependent output loss is the reason calibration will matter.

## bridge {hold=1.5 transition=fade}
[#a] Attention, normalization, nonlinear paths and residual additions can redistribute error. [#b] Some effects cancel while others grow or change the leading token margin. [#c] A different token then changes later generation history. [#d] Retain layer reconstruction diagnostics beside question-level regression cases. [#e] Do not infer grounded-answer preservation from weight mean squared error. [#f] Next we identify real converted artifacts and control the comparison factors.


---

# 11 · Compare identifiable weight artifacts

## question {hold=1.5 transition=fade}
[#a] Our real comparison will preserve a floating reference and derive eight-bit and four-bit artifacts. [#b] Each must come from the same frozen source. [#c] Conversion changes the runtime container, while quantization changes numerical representation. [#d] Keep those steps and their receipts separate. [#e] A label such as Q four K M can describe a mixed tensor preset. [#f] Inspect the saved artifact rather than treating its name as exact whole-model precision.

## reasoning {hold=1.5 transition=fade}
[#a] Keep source weights, tokenizer and complete prompt identity unchanged. [#b] Also fix runtime build, placement, cache policy and decoding. [#c] Input and output caps, question membership, batch and concurrency belong in the protocol. [#d] The weight-only candidate changes the artifact and its declared recipe. [#e] Changing placement or tokenizer creates additional possible causes. [#f] The reference checker reports those changed factors explicitly.

## code-11-artifact-plan-py {hold=1.5 transition=fade min=32}
[#p0] The controlled-factor tuple lists identities and execution choices that must match. These fields include tokenizer, prompt, runtime, placement, cache, budgets and question membership. [#p1] Changed-factors compares each declared control. Main constructs a baseline with explicit fixture identities and unchanged execution settings. [#p2] The first candidate changes only its weight-format label and passes. The second changes placement and tokenizer, so both mismatches must appear in the returned list. [#p3] The final record classifies the result as a deployment comparison. Metadata checks expose confounds, while actual commands and run receipts still need to establish the claimed settings.

## result {hold=1.5 transition=fade}
[#a] The first fixture candidate changes only its weight-format field. [#b] It has no mismatched controlled factors. [#c] The second changes tokenizer identity and placement. [#d] The checker returns both differences and classifies a deployment comparison. [#e] Quality on matched questions may remain useful under that deployment design. [#f] A timing difference can no longer isolate weight compression alone.

## transfer {hold=1.5 transition=fade}
[#a] Suppose the floating artifact fits only on the CPU while Q four fits on a device. [#b] Keep the floating answers as a documented quality anchor. [#c] Report each timing row with its own placement and complete runtime settings. [#d] Do not call their latency ratio a quantization-only speedup. [#e] A separate same-placement comparison is needed for that narrower mechanism claim. [#f] Retain out-of-memory and unsupported outcomes rather than dropping them.

## bridge {hold=1.5 transition=fade}
[#a] Preserve source hashes, commands, warnings and destination artifact hashes. [#b] Check requested operations against actual tensor coverage. [#c] A bounded pilot verifies prompt agreement, record completeness and reload. [#d] Then freeze the declared development workload before the main run. [#e] Conditional quantizer commands in the written lesson remain future implementation instructions. [#f] Next we measure which weight errors calibration inputs amplify.


---

# 12 · Measure calibration sensitivity

## question {hold=1.5 transition=fade}
[#a] Consider two row-error vectors of the same size. [#b] Candidate A changes the first weight by one quarter. [#c] Candidate B changes the second weight by one quarter. [#d] Both have mean squared weight error zero point zero three one two five. [#e] Our calibration inputs are four and one, then four and minus one. [#f] Those inputs amplify the first-channel error much more strongly.

## reasoning {hold=1.5 transition=fade}
[#a] For each sample, take the error-vector dot product with the input and square it. [#b] Average those squares over the sample count. [#c] Expanding the products groups the input cross-products into a Gram matrix. [#d] Our diagonal entries are sixteen and one, while off-diagonal terms cancel. [#e] Candidate A therefore has output mean squared error one. [#f] Candidate B has zero point zero six two five, a sixteenfold difference in this fixture.

## code-12-calibration-py {hold=1.5 transition=fade min=32}
[#p0] Gram rejects empty and ragged matrices before defining the input dimension. It then computes uncentered cross-products averaged over the actual sample count. [#p1] The quadratic helper expands all channel pairs. Reconstruction independently computes each sample output error directly, providing a separate route to the same loss. [#p2] Main forms the two-channel fixture and checks its exact Gram matrix. Both candidate errors are then checked against their known direct and quadratic losses. [#p3] The final JSON shows equal weight-error size with unequal output mean squared errors. No activation hooks or model calibration run is performed by this reference.

## result {hold=1.5 transition=fade}
[#a] The complete reference first constructs the uncentered second-moment matrix. [#b] It evaluates the quadratic loss for each error vector. [#c] It separately computes every dot-product output error and averages its square. [#d] The direct and expanded calculations agree with the known numbers. [#e] Empty or ragged sample matrices are rejected. [#f] A real collector must also preserve layer axes, valid-token masks and sample counts.

## transfer {hold=1.5 transition=fade}
[#a] Duplicate every calibration sample once. [#b] The sum of cross-products doubles, but the sample count doubles too. [#c] The Gram matrix and average reconstruction loss therefore stay unchanged. [#d] A sum-based implementation would incorrectly double them. [#e] Duplicated observations do not create new independent evidence or new input directions. [#f] More stored tokens are not automatically more representative calibration.

## bridge {hold=1.5 transition=fade}
[#a] Compare general text and document-domain text with the same declared token budget. [#b] Keep method, group size, precision, seed and runtime controlled. [#c] A runtime-native importance file has its own format and coverage checks. [#d] The toy Gram JSON is not a drop-in collector artifact. [#e] Layer reconstruction loss remains separate from grounded-answer quality. [#f] Next we rescale channels while preserving the original floating computation.


---

# 13 · Understand activation-aware scaling

## question {hold=1.5 transition=fade}
[#a] Calibration showed that a large input channel amplifies weight error. [#b] We can scale weight columns while inversely scaling input channels. [#c] Those two changes preserve the original floating linear operation. [#d] The quantizer can nevertheless choose different reconstruction levels afterward. [#e] This creates an opportunity to reduce output error. [#f] The improvement must be checked because shared group scales can hurt other channels.

## reasoning {hold=1.5 transition=fade}
[#a] Weights zero point two four and one point zero two multiply input four and one. [#b] Their exact output is one point nine eight. [#c] Scale the first channel by four to obtain weights zero point nine six and one point zero two. [#d] The inversely scaled input is one and one, preserving that output. [#e] Our toy ternary grid reconstructs both scaled weights as one point zero two. [#f] Its output is 2.04, so the error above 1.98 is 0.06.

## code-13-activation-scaling-py {hold=1.5 transition=fade min=24}
[#p0] Dot and transformed are defined first. The transformation scales each weight channel and inversely scales its input, before ternary reconstruction defines the coarse teaching grid. [#p1] Loss checks floating equivalence for every sample, then evaluates reconstructed output error. Main searches the four declared scales and asserts every expected loss. [#p2] The printed table identifies scale four as this fixture winner. Its explicit ternary label prevents treating the result as a real four-bit exporter or task-quality measurement.

## result {hold=1.5 transition=fade}
[#a] Across the two calibration inputs, no scaling gives loss zero point nine two one six. [#b] Scale two gives the same loss in this fixture. [#c] Scale four reduces it to zero point zero zero three six. [#d] Scale eight increases loss to zero point eight one. [#e] The program verifies floating equivalence before evaluating every quantized candidate. [#f] These are authored ternary-grid numbers, not a measured four-bit model result.

## transfer {hold=1.5 transition=fade}
[#a] Keep the transformed weights zero point nine six and one point zero two. [#b] Now accidentally keep the old input four and one. [#c] The output becomes four point eight six instead of one point nine eight. [#d] The original floating function has changed before quantization even begins. [#e] A graph rewrite must therefore verify both sides of the equivalent transform. [#f] Branches and shared consumers can constrain where those scales are absorbed.

## bridge {hold=1.5 transition=fade}
[#a] {AWQ|A W Q} uses activation information and equivalent scaling to protect salient channels. [#b] Its full implementation includes architecture-aware transformations, search and quantizer integration. [#c] Our reference isolates the mechanism without implementing a production exporter. [#d] Inspect actual completed layer coverage and any skipped operations. [#e] A requested method flag does not establish that the selected model received that method. [#f] Next we explain how correlated channels can compensate for a rounding error.


---

# 14 · Derive a compensation step

## question {hold=1.5 transition=fade}
[#a] Suppose the first weight has already been forced onto a quantization grid. [#b] Its error is now fixed while another weight can still change. [#c] Correlated inputs mean the two errors can partly cancel in the layer output. [#d] We express that interaction as a quadratic reconstruction loss. [#e] Then we solve one continuous constrained step. [#f] This explains the mechanism without claiming a globally optimal discrete model.

## reasoning {hold=1.5 transition=fade}
[#a] Use the symmetric Gram matrix with rows four, two and two, two. [#b] Its quadratic loss contains the squared first error, the cross term and the squared second error. [#c] Fix the first error at one quarter and differentiate with respect to the second. [#d] The derivative is one plus four times the second error. [#e] Setting it to zero gives minus one quarter. [#f] A positive second derivative confirms the unique minimum along that coordinate.

## code-14-compensation-py {hold=1.5 transition=fade min=24}
[#p0] Cost expands the two-channel quadratic. Main fixes the first error, calculates the analytic compensation and independently searches a dense scalar grid for the minimum. [#p1] The finite-difference slope checks the stationary point while another assertion requires a lower loss than no compensation. The visible output retains both costs. [#p2] The final record and entry point complete the constrained-step demonstration. Production damping, ordering, quantizer blocks and model export are deliberately outside this small program.

## result {hold=1.5 transition=fade}
[#a] Without compensation, the fixture loss is one quarter. [#b] Substituting the negative one-quarter compensation reduces it to one eighth. [#c] The reference also searches two thousand one scalar candidates independently. [#d] That grid includes the analytic optimum and finds the same value. [#e] A finite-difference slope near the optimum is approximately zero. [#f] These checks catch a sign error or a missing factor in the cross term.

## transfer {hold=1.5 transition=fade}
[#a] Remove the two off-diagonal Gram entries. [#b] The matrix now has only its diagonal values of four and two. [#c] With the first error fixed, any second error increases the loss. [#d] Its best value is therefore zero. [#e] Large diagonal sensitivity alone does not supply a useful correction direction. [#f] The cross term determines how the two errors can compensate.

## bridge {hold=1.5 transition=fade}
[#a] {GPTQ|G P T Q} uses layerwise reconstruction and approximate second-order information. [#b] Its practical algorithm adds damping, ordered updates, blocks and quantizer machinery. [#c] Damping stabilizes poorly observed directions while changing the chosen objective. [#d] Later rounding of the compensating coordinate can lose part of the continuous benefit. [#e] Our source does not establish a supported Gemma export. [#f] Next we choose precision and group size under an explicit storage cap.


---

# 15 · Allocate precision under a fixed budget

## question {hold=1.5 transition=fade}
[#a] Different layers can have different sensitivity to reconstruction errors. [#b] A smaller group adds scale metadata while allowing a more local range. [#c] Higher precision increases payload and may require another kernel. [#d] We need to filter by feasibility before choosing the smallest loss. [#e] The two-layer fixture is small enough to enumerate every option. [#f] Its authored loss table is a diagnostic proxy rather than measured model quality.

## reasoning {hold=1.5 transition=fade}
[#a] At 4 bits with groups of 32, the 1024 value layer costs 576 bytes. [#b] The 256 value layer costs 144 bytes with the same representation. [#c] Using 8 bits for that smaller layer costs 272 bytes. [#d] Together the chosen layers need 848 bytes. [#e] This satisfies the 900 byte cap. [#f] Using 8 bits for both layers would need 1360 bytes and fail.

## code-15-precision-budget-py {hold=1.5 transition=fade min=24}
[#p0] Storage counts payload and group-scale overhead separately. Main declares the allowed bit/group choices and the two authored per-layer loss tables. [#p1] Every recipe pair is enumerated with byte cost and proxy loss. The feasibility filter applies the nine-hundred-byte cap before selecting the minimum-loss candidate. [#p2] The known winner, eight hundred forty-eight bytes, is asserted before printing. The record preserves recipe, loss and budget so the decision can be independently checked.

## result {hold=1.5 transition=fade}
[#a] The large layer contributes proxy loss zero point zero eight at four bits. [#b] The small sensitive layer contributes zero point zero two at eight bits. [#c] Their total is zero point one at eight hundred forty-eight bytes. [#d] The enumeration filters candidates over budget before minimizing that loss. [#e] An additive layer-loss proxy still omits interactions in the full network. [#f] The selected real recipe therefore needs task and runtime evidence.

## transfer {hold=1.5 transition=fade}
[#a] Lower the capacity to 700 bytes; the previous 848 byte recipe now fails. [#b] Both layers at 4 bits with groups of 32 would still need 720 bytes. [#c] Use group 128 for the large layer and group 32 for the smaller layer. [#d] This costs 528 plus 144, or 672 bytes. [#e] The authored proxy loss is 0.42. [#f] Keep the original feasibility rule and reject recipes above the new cap.

## bridge {hold=1.5 transition=fade}
[#a] A candidate can save storage yet lose support quality or become slower. [#b] Nondominated choices still need an operational decision rule. [#c] For example, enforce memory, quality and reload constraints before minimizing latency. [#d] Record actual tensor composition, kernel support and measured peak memory. [#e] If no candidate passes, report no selection. [#f] Next we score answers, citations, refusals and failures with explicit denominators.


---

# 16 · Score grounded answers and refusals

## question {hold=1.5 transition=fade}
[#a] The assistant promises supported answers from supplied material. [#b] A wrong fact, an unrelated citation and a false refusal are different failures. [#c] A truncated guessed warranty can be both unsupported and incomplete. [#d] Keep each outcome and its raw response visible. [#e] Before scoring, every candidate must match the same question identifiers exactly once. [#f] Missing or duplicate predictions cannot silently improve the denominator.

## reasoning {hold=1.5 transition=fade}
[#a] Our fixture has two answerable questions and one unanswerable warranty question. [#b] The charging answer is correct and supported. [#c] The adapter-voltage question receives a false refusal. [#d] The warranty response guesses an absent fact and reaches the output cap. [#e] Supported-answer rate is one out of two, while unsupported-answer rate over all questions is one out of three. [#f] The same truncation can count in another failure category without becoming another question.

## code-16-evaluator-py {hold=1.5 transition=fade min=32}
[#p0] Evaluate first checks exact question membership. Missing, extra or duplicate prediction identifiers raise an error before scoring. [#p1] The counters separate question membership and outcome categories. Each matched prediction declares refusal state and finish reason. [#p2] For answerable questions, accepted aliases check correctness. Required evidence checks this fixture support, while false refusals and unsupported attempts have separate counters. [#p3] For unavailable facts, the final branch counts correct refusals or unsupported attempts. The complete evaluator then returns its counters.

## code-16-fixture-py {hold=1.5 transition=fade min=40}
[#p0] Main now declares the reviewed gold questions. The charging duration and adapter voltage are answerable. [#p1] The warranty question has no supplied answer. The fixture then begins responses with explicit question identifiers. [#p2] The first response is a supported charging answer. The next falsely refuses the known adapter voltage, using the declared empty-answer refusal schema. [#p3] The third response invents a warranty and reaches the length limit. All three records enter the previously defined evaluator under fixed membership. [#p4] Assertions require one supported answer, one false refusal and one unsupported answer. The printed record also retains truncation. The entry point completes the program.

## result {hold=1.5 transition=fade}
[#a] The reference checks exact accepted answer aliases and required evidence identifiers. [#b] Its counts expose the supported answer, false refusal, unsupported guess and truncation. [#c] A real evaluator also needs schema checks and reviewed semantic support. [#d] An existing passage identifier does not establish that every claim follows from it. [#e] Alternative valid evidence sets and multi-fact completeness need explicit labels. [#f] Retain reviewer disagreements instead of treating model confidence as ground truth.

## transfer {hold=1.5 transition=fade}
[#a] Suppose two of the frozen questions crash during generation. [#b] Do not remove those identifiers and score only the easy completed remainder. [#c] Reject the incomplete membership or retain explicit failures under a declared policy. [#d] A partial score can be reported as partial with completed and expected counts. [#e] The main matched denominator must remain visible. [#f] Otherwise a less reliable candidate could appear more accurate by failing difficult inputs.

## bridge {hold=1.5 transition=fade}
[#a] Compare candidate and reference support indicators on the same question. [#b] Their difference is minus one, zero or plus one. [#c] An improvement and a regression can cancel in the mean while affecting different facts. [#d] Review slices such as input length, evidence position and numerical answers. [#e] Small slices need visible counts and uncertainty. [#f] Next we define latency intervals and actual token work with equal care.


---

# 17 · Measure performance with declared boundaries

## question {hold=1.5 transition=fade}
[#a] A fast warm decode is not the same measurement as a fast complete request. [#b] We distinguish user arrival, runtime submission, first token and last token. [#c] Prompt processing and generated-token intervals are separate operations. [#d] The boundary must be observable under the chosen runtime. [#e] A buffered network chunk may contain several tokens. [#f] Call the metric a chunk observation unless exact token events are available.

## reasoning {hold=1.5 transition=fade}
[#a] Let request arrival be zero and runtime submission zero point two seconds. [#b] The first token arrives at zero point seven and the last at two point seven. [#c] User time to first token is zero point seven, while runtime time is zero point five. [#d] Five tokens create four intervals after the first token. [#e] Four divided by two seconds gives two decode tokens per second. [#f] An independently measured hundred-token prefill in zero point four seconds gives two hundred fifty tokens per second.

## code-17-timing-py {hold=1.5 transition=fade min=32}
[#p0] Summarize validates temporal ordering, at least two generated tokens and a positive separately measured prefill interval. Invalid timing contracts fail before a rate is returned. [#p1] Each returned field names its duration or work boundary. Nearest-rank then defines its percentile estimator and rejects empty samples or invalid probability. [#p2] Main supplies invented timestamps and checks the known decode and prefill rates. The output also records that these values were not measured from a model. [#p3] The closing output and entry point preserve that explicit scope. Actual backend synchronization, cold-state control and repeated measurement remain part of the real protocol.

## result {hold=1.5 transition=fade}
[#a] The reference rejects invalid timestamp order and unsuitable token counts. [#b] It computes separate end-to-end, first-token, prefill and decode quantities. [#c] Its percentile helper uses a declared nearest-rank estimator. [#d] With only five observations, a reported tail percentile is unstable. [#e] Retain individual samples, generated counts and finish reasons. [#f] Every displayed time here is an arithmetic fixture, not a measured model speed.

## transfer {hold=1.5 transition=fade}
[#a] A rate of five divided by two would give two point five tokens per second. [#b] That does not match the defined interval from first token to last token. [#c] The first token is already present at the start. [#d] Only four token arrivals remain within those two seconds. [#e] For one generated token, this inter-token rate has no interval and is undefined. [#f] A different whole-runtime rate can be useful when named with its own boundary.

## bridge {hold=1.5 transition=fade}
[#a] A host call may return before asynchronous device work has finished. [#b] Use verified synchronization or backend event timing for the measured operation. [#c] Record cold loading separately from warm requests and reset sequence state. [#d] Fix prompts, output policy, placement, threads, batch and concurrency. [#e] Alternate or randomize run order under a frozen bounded protocol. [#f] Next we hold weights fixed and derive how cache state grows with context.


---

# 18 · Derive KV-cache growth

## question {hold=1.5 transition=fade}
[#a] Attention retains prior keys and values to avoid recomputing the full history. [#b] That saved state depends on sequence length and request ownership. [#c] The logical tensor axes include batch, key-value heads, retained tokens and head dimension. [#d] Grouped-query attention can use fewer key-value heads than query heads. [#e] A local window can stop retaining old positions. [#f] Shared cache owners must be counted once rather than once per consuming layer.

## reasoning {hold=1.5 transition=fade}
[#a] Two layers in this authored fixture share one sliding cache owner with window 512. [#b] That owner has 2 key-value heads of dimension 64. [#c] A third layer owns a full-context cache with 2 heads of dimension 128. [#d] At sequence length 2048 and batch 1, each scalar uses 2 bytes. [#e] The sliding owner needs 262144 bytes and the full owner needs 2097152 bytes. [#f] Their total is 2359296 bytes.

## code-18-owner-py {hold=1.5 transition=fade min=16}
[#p0] Cache bytes visits layers and records each owner once. A repeated owner is skipped. The sliding window limits retained positions. [#p1] The product counts keys and values, batch, retained tokens, key-value heads, head dimension and scalar bytes. Return their total.

## code-18-fixture-py {hold=1.5 transition=fade min=16}
[#p0] Main declares one shared sliding owner and one full owner. The known length totals and doubled-batch assertion verify this authored fixture. [#p1] The JSON labels deduplicated logical cache bytes. Actual padding, metadata, allocation and Gemma ownership must be checked separately. The entry point only runs this fixture.

## result {hold=1.5 transition=fade}
[#a] Evaluate lengths 512, 2048 and 4096. [#b] The sliding allocation stops growing when its window is full. [#c] The full-history term continues growing. [#d] Count the shared sliding owner only once. [#e] These are logical scalar bytes before padding, scales, residuals or allocator reservation. [#f] This authored fixture does not claim to describe the actual Gemma cache owners.

## transfer {hold=1.5 transition=fade}
[#a] At length four thousand ninety-six, double batch from one to two. [#b] Under this independent-sequence fixture, both owner allocations double. [#c] Total logical storage becomes eight million nine hundred twelve thousand eight hundred ninety-six bytes. [#d] The sliding owner still retains five hundred twelve positions per sequence. [#e] Unequal-length batches or paged allocations need their actual storage rules. [#f] Physical reserved memory may stay constant even while logical used tokens grow.

## bridge {hold=1.5 transition=fade}
[#a] The model metadata includes hybrid attention and cache-sharing information. [#b] The runtime determines actual owners, layout, precision and reservation policy. [#c] Inspect that mapping instead of multiplying every layer by one uniform dimension. [#d] Freeze one weight artifact before the cache-policy study. [#e] Measure memory, quality and latency at bounded prompt lengths with output allowance fixed. [#f] Next we isolate how key and value quantization errors affect attention differently.


---

# 19 · Quantize cache without confounding the weight study

## question {hold=1.5 transition=fade}
[#a] Hold the chosen weight artifact fixed before changing cache precision. [#b] An attention key helps determine how much probability a position receives. [#c] A value contains information mixed using those probabilities. [#d] Changing a key can redistribute the whole probability vector. [#e] Changing only a value leaves the existing probabilities unchanged. [#f] We isolate those two pathways in a scalar-head example.

## reasoning {hold=1.5 transition=fade}
[#a] Use query one, keys zero and logarithm of three, and values zero and four. [#b] The probabilities are one quarter and three quarters, so output is three. [#c] Add another logarithm of three to the second key. [#d] Relative weights become one and nine, giving output three point six. [#e] Instead restore the keys and increase only the second value to five. [#f] Probabilities stay unchanged and output becomes three point seven five.

## code-19-cache-error-py {hold=1.5 transition=fade min=24}
[#p0] Attend forms scores, normalizes their exponentials and mixes values. It returns both output and probabilities so a value-only change can check unchanged attention weights. [#p1] Main evaluates the base case and independent key and value changes. It asserts the expected outputs and the unchanged probability vector for the value perturbation. [#p2] The final JSON exposes all three outputs and the original mixture. No packed cache, kernel or architecture-specific allocation is hidden behind this scalar function.

## result {hold=1.5 transition=fade}
[#a] The complete reference computes each attention result from supplied scalar data. [#b] Its value-only case asserts that probabilities remain identical. [#c] The key case recomputes normalized scores before mixing values. [#d] This tests the mathematical pathways rather than any packed-cache implementation. [#e] {KIVI|key vee} motivates distinct key and value quantization axes and a high-precision residual region. [#f] Its full algorithm, model support and device requirements remain separate implementation questions.

## transfer {hold=1.5 transition=fade}
[#a] Keep probabilities one quarter and three quarters. [#b] Change values from zero and four to two and four. [#c] Output becomes one half plus three, totaling three point five. [#d] The output shift is one quarter times the two-unit value change. [#e] The probabilities stay fixed because the keys and query were not changed. [#f] A key change would require a different calculation through score normalization.

## bridge {hold=1.5 transition=fade}
[#a] A two-bit older cache with recent high-precision tokens has more than two bits of effective storage. [#b] Include residual regions, scale metadata, padding and runtime workspaces. [#c] Record actual applied key and value policies, axes and layer coverage. [#d] Test evidence position, distractors and questions requiring distant facts. [#e] An unsupported or silently floating policy remains an explicit failed comparison. [#f] Next we save artifact generations that can be verified after restart.


---

# 20 · Save and recover a verified generation

## question {hold=1.5 transition=fade}
[#a] A process can answer once while its saved files are incomplete or incompatible. [#b] Identity says which source and recipe the artifact represents. [#c] Integrity checks whether bytes match their recorded size and digest. [#d] Compatibility asks whether the selected runtime interprets those bytes correctly. [#e] A hash match alone establishes neither provenance truth nor answer quality. [#f] We save immutable generations and disclose any fallback.

## reasoning {hold=1.5 transition=fade}
[#a] Write a new generation into a partial staging directory. [#b] Complete and verify its artifact files and manifest. [#c] Then rename the directory on the same filesystem to its immutable generation name. [#d] Readers ignore partial directories and validate complete ones newest first. [#e] A corrupt newest generation can fall back to a verified older generation. [#f] Atomic visibility is different from crash durability, which requires its own storage policy.

## code-20-recovery-py {hold=1.5 transition=fade min=48}
[#p0] Publish creates a unique partial directory and writes its toy payload. Its manifest binds generation name, expected filename, size and digest. [#p1] The completed directory is renamed on the same filesystem. Load-latest scans complete generation names newest first and ignores partial staging directories. [#p2] Each candidate manifest is checked for schema, generation identity and the fixed expected path. The loader reads its bytes only after those identity checks. [#p3] Size and digest mismatch rejects a generation, while a valid one returns with rejection history. Main then publishes two temporary generations for the failure probe. [#p4] The newest payload is deliberately corrupted and a partial newer directory is left behind. The loader must fall back to verified generation one. [#p5] The printed result exposes loaded and rejected generations and the partial-directory decision. The temporary-directory lifetime keeps this reference from modifying real artifacts.

## result {hold=1.5 transition=fade}
[#a] The reference writes tiny toy weight bytes inside a temporary directory. [#b] It publishes two complete generations, then corrupts the second. [#c] It also leaves a third directory marked partial. [#d] The loader must return generation one and retain the rejected generation identity. [#e] A separate-process audit also reloads the toy bytes without in-memory state. [#f] This demonstrates integrity and fallback, not production durable model storage.

## transfer {hold=1.5 transition=fade}
[#a] Now suppose the newest complete generation is corrupt and no valid older generation exists. [#b] The correct result is no verified generation available. [#c] Do not relabel the corrupt bytes as loaded. [#d] Do not silently substitute another source model under the chosen candidate name. [#e] Retain each rejection reason and expose the failure to the application. [#f] Recovery preserves the contract rather than manufacturing a plausible answer.

## bridge {hold=1.5 transition=fade}
[#a] The eventual model manifest also binds tokenizer, template, runtime and accepted cache settings. [#b] Exit the process before the recovery acceptance test. [#c] Reload only retained verified artifacts and replay the frozen questions. [#d] Declare whether acceptance requires identical tokens or an explicitly reviewed output contract. [#e] Record the actual loaded generation and any fallback. [#f] Next we give inference a request-owned lifecycle and response boundary.


---

# 21 · Own request state and validate the answer boundary

## question {hold=1.5 transition=fade}
[#a] A running adapter can retain an active sequence and its key-value state. [#b] An independent document question needs a fresh request-owned lifecycle. [#c] We define request data, begin the sequence, generate, validate and reset. [#d] Cleanup must also occur when response validation fails. [#e] The current adapter deliberately returns a fixed fixture response. [#f] It exercises control flow without claiming to infer or retrieve an answer.

## reasoning {hold=1.5 transition=fade}
[#a] An answered response needs text and at least one citation. [#b] A not-in-document response uses empty answer text and empty citations under this schema. [#c] Every cited identifier must belong to the passages supplied in this request. [#d] These checks reject malformed or unknown evidence references. [#e] They do not prove that the cited wording entails the claim. [#f] The semantic support rubric from the opening lesson is still required.

## code-21-part-1 {hold=1.5 transition=fade min=16}
[#p0] The request dataclass declares question and supplied passages. Fixture-adapter starts with explicit active-state and reset counters; it contains no model runtime. [#p1] Begin rejects reuse of an already-active session. Generate requires an active request and constructs a fixed response whose fixture backend will remain visible.

## code-21-part-2 {hold=1.5 transition=fade min=16}
[#p0] Reset clears the active flag and counts cleanup. Answer validates the input budget and evidence before beginning and generating inside its protected lifecycle. [#p1] The response field set and citation-list type are checked before accepting status. Malformed shape or an unsupported answer status raises a validation error.

## code-21-part-3 {hold=1.5 transition=fade min=16}
[#p0] An answered result requires text and citations. A refusal must leave both empty under this schema; the conditions distinguish structural statuses before checking evidence membership. [#p1] Unknown citations raise, while finally always resets the request-owned fixture state. Main constructs the known charging request and runs the first independent call.

## code-21-part-4 {hold=1.5 transition=fade min=16}
[#p0] A second call must reproduce the fixture response with another reset. The third request supplies an unrelated evidence identifier to force citation validation to fail. [#p1] The exception is retained as expected rejection, and the reset count must reach three. A missed cleanup path or accepted invalid citation would fail these assertions.

## code-21-part-5 {hold=1.5 transition=fade min=8}
[#p0] The output records fixture backend, reset count and the rejection result. The visible lifecycle does not establish semantic support or a real backend cancellation contract.

## result {hold=1.5 transition=fade}
[#a] The first two fixture calls return the same response in separate sessions. [#b] A third request supplies only an unrelated passage identifier. [#c] The adapter still returns its fixed charging citation. [#d] Validation rejects that identifier and the finally block performs cleanup. [#e] Three resets are observed and the adapter is no longer active. [#f] The exercised path does not prove all real backend cancellation or resource behavior.

## transfer {hold=1.5 transition=fade}
[#a] Suppose a malformed response raises while checking citations. [#b] Release the request-owned state in finally and retain the error. [#c] Do not fabricate a not-in-document answer to hide that failure. [#d] A failure during begin can involve a different already-active request. [#e] Concurrent code therefore needs ownership-aware handles rather than a global reset. [#f] The real adapter also needs timeout, cancellation and partial-stream checks.

## bridge {hold=1.5 transition=fade}
[#a] The eventual adapter renders the pinned template and counts complete prompt tokens. [#b] It rejects excessive input before silently dropping evidence. [#c] It binds the verified artifact, placement and supported cache policy. [#d] Retain raw outputs, token identifiers, finish reason and measurements. [#e] A later retriever must identify both retrieved passages and those actually packed. [#f] Next we connect a local interface with visible backend and artifact provenance.


---

# 22 · Connect a truthful local interface

## question {hold=1.5 transition=fade}
[#a] An interface can return a polished answer without running any model. [#b] Our reference therefore accepts only one named teaching fixture. [#c] It reports backend fixture and model executed false in its response. [#d] That narrow contract makes the demonstration scope testable. [#e] The future application needs arbitrary supported document requests through a real adapter. [#f] Its backend, artifact generation and context limits must remain visible.

## reasoning {hold=1.5 transition=fade}
[#a] Bind a local HTTP server to an operating-system-selected free port. [#b] The answer route checks the body size and parses a JSON object. [#c] Only the named Lumen charging fixture is accepted. [#d] The response states content type, content length and explicit fixture identity. [#e] The test submits both a valid fixture and an invalid request. [#f] Finally it shuts down the server, closes the socket and joins the thread.

## code-22-local-app-py {hold=1.5 transition=fade min=56}
[#p0] The handler imports only standard-library transport and concurrency tools. Respond serializes a JSON body and begins an explicit content-type and byte-length response. [#p1] The answer route rejects unknown paths, reads the declared body length and enforces a bounded request size. This is the narrow fixture API. [#p2] Parsed content must equal the named charging fixture. Invalid input receives an explicit bad-request response carrying the fixture backend identity. [#p3] Success also reports fixture identity and model-executed false. The server handler therefore does not pretend arbitrary document questions have been answered. [#p4] Main starts a loopback server on a free port, launches its thread and submits the valid fixture request with a bounded client timeout. [#p5] The success response is parsed, then an invalid request must raise the expected bad-request status. The explicit no-model flag is checked as part of the contract. [#p6] The final JSON exposes both statuses and the returned fixture. Finally shuts down the server, closes the socket and joins its thread before the program exits.

## result {hold=1.5 transition=fade}
[#a] The valid request returns three hours with the charging passage identifier. [#b] The invalid fixture request receives a bad-request status. [#c] The test also verifies that no model was executed. [#d] A successful HTTP response alone proves only that a handler replied. [#e] This server does not implement streaming, arbitrary document QA or a browser interface. [#f] Those future capabilities need their own actual interaction evidence.

## transfer {hold=1.5 transition=fade}
[#a] Suppose the HTTP demo starts again and returns its fixed answer. [#b] That shows the fixture server can run again. [#c] It does not establish that a saved Gemma artifact was loaded. [#d] Real acceptance must verify the chosen generation, document evidence and adapter output after restart. [#e] The interface must also expose fallback and document changes. [#f] Keep transport checks separate from model-backed application evidence.

## bridge {hold=1.5 transition=fade}
[#a] The current TUI remains available for status, setup checks and configuration. [#b] Add an application runner after the real service passes its acceptance contract. [#c] It should use the same canonical entry point as the CLI. [#d] Test supported and unavailable facts, citations, document changes, cancellation and malformed inputs. [#e] A failed process must not remain labeled running. [#f] Next we freeze development selection before opening final results.


---

# 23 · Select on development and seal the final test

## question {hold=1.5 transition=fade}
[#a] We can vary recipes, calibration, grouping, context and thresholds during development. [#b] Choosing repeatedly against final answers adapts the choice to that assessment. [#c] We instead freeze a decision rule before opening final labels. [#d] Feasibility combines memory, support quality and verified reload. [#e] Only passing candidates compete on the declared latency objective. [#f] A truthful decision can be that no candidate satisfies the contract.

## reasoning {hold=1.5 transition=fade}
[#a] All shown memory values use gibibytes. The cap is 8 and minimum support is 0.92. [#b] The Q8 candidate scores 0.95 but needs 9. [#c] The Q4 candidate fits at 6 but scores 0.90. [#d] The first fails memory and the second fails quality. [#e] Neither can be selected under the original rule. [#f] Relaxing a threshold after seeing these values creates a new decision version.

## code-23-part-1 {hold=1.5 transition=fade min=16}
[#p0] Choose filters memory, answer support and verified reload. Passing candidates are ordered by latency and a deterministic name. [#p1] No feasible candidate returns no selection. The bootstrap helper builds every ordered resample of the fixed paired differences.

## code-23-part-2 {hold=1.5 transition=fade min=16}
[#p0] Main supplies two candidate configurations. Their memory, quality and timing fields are authored examples. [#p1] Each candidate fails a different condition. The assertion requires no winner before declaring the paired differences.

## code-23-part-3 {hold=1.5 transition=fade min=16}
[#p0] Enumeration creates 256 resamples with original mean zero. The endpoints describe their full range, not confidence limits. [#p1] The final record retains both examples. Real document groups and a sealed final test need a separate protocol.

## result {hold=1.5 transition=fade}
[#a] Use paired support differences zero, minus one, zero and plus one. [#b] Their average is zero, but one regression and one improvement remain. [#c] The tiny example enumerates all four to the fourth ordered bootstrap resamples. [#d] That produces two hundred fifty-six possible resampled means. [#e] Their full range is minus one to plus one and is not a confidence interval. [#f] Real uncertainty must respect documents shared by related questions.

## transfer {hold=1.5 transition=fade}
[#a] Suppose a shorter context or another model could satisfy the deployment budget. [#b] That may be a sensible next design, but it changes the protocol. [#c] Record the new source or context decision and repeat development selection. [#d] Keep the final labels untouched during that revision. [#e] Do not retroactively call the old failing candidate successful. [#f] Preserve the earlier negative result and its original thresholds.

## bridge {hold=1.5 transition=fade}
[#a] Freeze artifact, runtime, placement, tokenizer, template and cache policy. [#b] Also freeze packing, decoding, scoring rubric, question membership and report code. [#c] Run the selected candidate and declared reference on the complete final membership. [#d] Retain failures, paired outcomes, resource work and uncertainty method. [#e] A later repair needs fresh evidence or an explicit reused-test limitation. [#f] Next we connect every phase to a concrete evidence handoff.


---

# 24 · Review every phase and hand off honestly

## question {hold=1.5 transition=fade}
[#a] We began with a correct charging number attached to the wrong evidence. [#b] The full route now connects canonical text, membership, prompts, artifacts and generated output. [#c] It also connects scoring, memory, timing, selection and restart behavior. [#d] Each boundary can fail independently. [#e] A successful setup command or teaching example cannot satisfy every model milestone. [#f] The final handoff therefore asks for specific evidence rather than a general success label.

## reasoning {hold=1.5 transition=fade}
[#a] Phase zero needs compatibility and a measured feasible budget. [#b] Phase one needs reproducible documents, reviewed labels and frozen membership. [#c] Phases two and three need matched reference, eight-bit and four-bit artifacts and outcomes. [#d] Phase four needs completed calibration or advanced-method coverage and controlled comparisons. [#e] Phase five needs actual cache-policy and context evidence. [#f] Phase six needs a real saved application, recovery replay and frozen final assessment.

## code-24-handoff-py {hold=1.5 transition=fade min=24}
[#p0] The required evidence list names actual model, comparison, application and recovery milestones. A checked scaffold or teaching program does not replace any of these receipts. [#p1] Completion reports every missing verified milestone. Main supplies only the honest scaffold and teaching-reference state and invokes that evidence gate. [#p2] Assertions require the project to remain incomplete with ten missing categories. The output keeps phase zero and learner review pending instead of inventing progress.

## result {hold=1.5 transition=fade}
[#a] The reference recognizes that scaffold and teaching components are checked. [#b] It still lists ten missing model and application evidence categories. [#c] That is the expected result after tutorial and video preparation. [#d] No model weights, benchmark scores or hosted execution were produced by these references. [#e] Learner explanation also needs the learner own responses. [#f] The next real implementation phase remains compatibility and budget.

## transfer {hold=1.5 transition=fade}
[#a] Suppose every displayed code example and numerical audit passes. [#b] We can trust the scoped arithmetic and exercised component contracts. [#c] We still cannot claim that the selected model loads, answers or meets the memory budget. [#d] Those claims need executed model receipts under the frozen protocol. [#e] A rendered tutorial or playable video also does not establish learner mastery. [#f] Keep source, model, application, learner and hosted evidence as separate states.

## bridge {hold=1.5 transition=fade}
[#a] Review the explanation and changed-case questions before advancing implementation. [#b] For real model work, recheck support and freeze a bounded phase-zero probe. [#c] After it passes, implement the document contract and proceed one evidence gate at a time. [#d] The sibling semantic-search project can later supply a separately versioned retriever. [#e] The complete written course and these captioned companions remain available together. [#f] Every future result should point to the exact inputs, operations and limits that support it.
