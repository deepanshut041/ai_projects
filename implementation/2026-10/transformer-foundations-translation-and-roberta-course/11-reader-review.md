# Tutorial 1 topic reader verification

Status: **pass**. All 9 content tests and both independently isolated browser suites pass. The tested render is `f58f3421af7fd17be2ff4567f8fd31a58f1868f7d82d38dec2c21d70859e6e83`. The final render is `62c5b39c47375e0f8244eac76d66b0b68251f914ac2773945ab7b2cc3c4c7ea3`; an independent manifest-hash comparison shows that only `topic-verification.json` changed after the successful browser runs. All chapters, diagrams, metadata, code and compiled lesson remain byte-identical. The parent records the same final evidence-only change in `topic-final-verification.json`.

## Coverage

Each deployment (`/` and `/project_docs/`) checked 112 records, 13 catalog sources and all 69 tutorial chapters at 320, 390, 768 and 1440 pixels: 448 record/width checks per deployment. Tutorial 1 has 32 chapters, 29 diagrams, 64 native practice items, 3 scenario controls and 11 cases. All 24 original Course 1 chapter IDs remain reachable, and all 19 Tutorial 0 compatibility redirects pass.

Every record passes math parsing, image decoding, Markdown-source downloads and internal article links with zero page or HTTP errors. Chapter contents/current state, previous/next and section anchors pass; code, tables and display equations are keyboard-scrollable. Search, tutorial order, source filters, mobile menus and the JavaScript-disabled navigation fallback pass. All practice reveals/hides and scenario outputs pass; print events show all 64 Course 1 answers and hide them afterward.

Actual browser clicks downloaded both Markdown controls in each deployment; all 4 files match the built bytes and their hashes are saved. The suite verifies the print action, print CSS and an A4 PDF. Additional Course 1 Lesson 18 print checks verify the action, white page, hidden navigation, visible answers and a 6-page Letter PDF in both layouts.

## Evidence and visual review

The build/test copies are `/mnt/projects/ai/.ai/tutorial1-topic-reader/2026-10-13/root` and `/mnt/projects/ai/.ai/tutorial1-topic-reader/2026-10-13/prefix`. Each excludes generated caches, symlinks only the installed `website/node_modules`, owns its own `dist` and uses its own random-port static server. The shared managed `dist` was not rebuilt or read by these suites. Exact commands, times, logs, downloads and hashes are in `topic-reader-verification.json`; each suite retains 93 PNGs and one A4 PDF under its `docs/website/test-results`.

Sixteen selected screenshots are in `/mnt/projects/ai/.ai/tutorial1-topic-reader/2026-10-13/selected-captures`: embedding, Q/K/V attention, translation optimization, MLM corruption and QA head at desktop/mobile sizes, plus the 320 px recovery fix. `reuse-proof.json` verifies all 61 chapter/diagram hashes against the final render. The parent inspected five mobile views and embedding/QA desktop pages and reported no remaining visual blocker.

## Corrections and limits

The first pass found a long inline checkpoint equation extending a 320 px document to 358 px. The parent converted it to scrollable display math. The next pass found two visibility insertions in the complete course; the parent retained the detailed mask control and removed the objective preview duplicate. Both fixes pass the complete rerun. Earlier failed and superseded snapshots/logs remain preserved in the evidence directory. All agent-created browser sessions and servers are closed.

This verifies the prepared reader. Learner mastery, hosted Colab recovery, benchmark capability and publication remain outside this task. No website source, frozen lesson prose or learner state was changed by this reader review.
