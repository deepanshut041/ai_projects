# Syntax highlighting in both documentation sites

23 October 2026. The user authorized fixing code highlighting throughout Projects and Research Docs. Lead: ai-project-workflow, with the shared planning contract. This is renderer/UI work; no tutorial source, media pin, model or research runtime is changed.

Cause verified: both common MarkdownIt renderers had no highlight callback; only escaped language fences were emitted. Projects already wrapped fences for keyboard horizontal scrolling, while Research did not. All reading surfaces use these common renderers. Original unrelated Attention edits are preserved; Research Docs and CLI started clean.

Implementation: pinned highlight.js 11.12.0 in both independent npm packages, build-time explicit-language highlighting, identical dark/print token themes, and accessible Research code fences. Unknown/missing/plaintext languages and Mermaid/video control blocks retain literal content. No automatic language guessing, client-side syntax runtime or CDN is added.

The Bash grammar additionally recognizes common workspace executables and command flags, retaining string/comment handling. This colors actual Research commands such as `./ai` and `uv run` that the stock Bash grammar otherwise leaves plain. Generated run-configuration JSON uses the same highlighter and code container.

Complete: all 21 Projects and 30 Research unit tests, both ordinary builds and every native browser suite pass. The browser suites retain navigation, exact downloads, private/public separation, Mermaid, captioned video controls and responsive checks. Added regression tests cover fourteen declared languages/aliases, source preservation, quoted shell options, unknown/plain/control languages, HTML escaping and unchanged inline code.

Dedicated syntax browser checks pass at `/` and each repository prefix, at 1440px and 390px. They find highlighted code on 129 Projects pages and six Research pages, exercise actual notes/tutorial/implementation/experiment/JSON pages, and use explicit language specimens for code copying, focus, scrolling, long lines, inert HTML and print wrapping. Lowest observed specimen token contrast is 8.20:1 on screen and 5.56:1 in print, above the 4.5:1 check. Representative screenshots were visually reviewed. See [the selected verification receipt](06-verification.json); full captures/reports were retained in each site's ignored test-results/ until the subsequent requested source cleanup archived them before removal.

Both managed readers are refreshed and running at http://127.0.0.1:4273/project_docs/ and http://127.0.0.1:4274/research_docs/. Six actual managed-reader checks pass: Attention notes, Research commands and Research JSON, each at desktop/mobile widths. Attention retains all seven video players and 66 highlighted tokens; Research commands/JSON show 19/868 tokens. Math and page-overflow checks pass with no browser JavaScript errors. The Research source-artifact audit passes (264 source/text files, 191 Markdown, 52 critical files).

No required work remains for this request. Model code, source-pinned teaching/video text and unrelated Attention edits are preserved. Research scientific jobs and paused containers were not started; learner review and scientific results remain separate. No publication or commit was requested or performed.
