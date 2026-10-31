# Preserve the P00 teaching standard in the tutorial skill

Date: 2026-10-09. Status: complete — implemented and locally verified.

## Request and scope

The user wants future tutorials to reproduce the integrated P00 teaching approach. Update the existing canonical workspace developer-learning-tutorials skill, its relevant references/templates and starter generation. Keep its current name and both umbrella discovery links; do not replace the independently scoped installed Dashboard skill. This is tutorial-tool preparation, not model work or a new research study.

## Evidence and decisions

- P00 now has one continuous lesson route with concepts, chapter depth, prerequisite maths, tiny calculations, file-by-file implementation, checks and handoffs on the same pages. Project records own implementation, sources, reference runs and readiness. The user rejected having to alternate between theory chapters and build lessons.
- The current skill still says editable chapters, proposes an orientation chapter before actual teaching and sends defense work to whiteboard/. Its initializer creates chapters/orientation.md. Update these defaults while preserving schema compatibility and existing P00/P01 artifacts.
- Reuse the local skill's authoring, adaptive and visual contracts. Put the substantial P00 pattern and audit rubric in one focused reference; provide reusable lesson and project-record templates. Keep maths topic-dependent and retain source/proof limits.
- Change the initializer to create a small draft in lessons/ plus meaningful project records. A draft must not look like verified content, advance milestones, overwrite revisions or prepopulate a whole roadmap.

## Acceptance

Skill/frontmatter/UI metadata validate; relative links and resources resolve; existing rendered lessons remain byte-preserved; a disposable new-topic scaffold uses one route and linked project records; incomplete drafts fail content validation. Run affected CLI tests/TUI smoke and existing site checks as appropriate. Save a manual instruction walkthrough covering a new project tutorial, a chapter/lesson consolidation, a maths second pass, and live tutoring. Do not label the walkthrough as a measured learning result.

## Progress and verification

The canonical skill now links a focused P00 teaching standard and three reusable templates. Authoring, adaptive-learning and defense instructions agree on one integrated lesson route and a proper project area. The initializer creates lessons/01-start.md, implementation/readiness records, research.md and its manifest. It reads all templates before creating a revision; missing assets leave no partial draft. Draft markers prevent unfinished lesson or linked project templates from rendering, while preserving the previous valid output.

The isolated one-lesson weighted-mean exercise passes initialization, draft rejection, example execution and rendering. Inspection exposed P00-specific introduction text, a hardcoded Project 0 sidebar, empty practice headings and eleven-lesson assumptions in the shared reader/tests. These now follow each course's own identity, first page and contents; P00-specific preservation assertions remain scoped to P00. The sample exists only in an ignored, disposable docs copy and was not added to the real project catalog.

| Verification | Result and scope |
| --- | --- |
| Skill validators and discovery | Both tutorial/defense skills valid; 15 local resource links resolve; both umbrella links resolve to the canonical skill; UI description/default prompt checked. |
| CLI | 77 tests pass, including 16 focused learning tests; TUI smoke still offers p00, project-docs and research-docs. New tests cover draft layout/reachability, missing-template atomicity and rejection of unfinished content with otherwise complete metadata. |
| Existing artifacts | All 66 preexisting P00/P01 tutorial files match the before-edit hashes; both packages still validate. |
| Reader native tests | 9 pass, including root/nested routes, one-lesson navigation and a project-specific identity without empty practice navigation. |
| Browser, normal site | 37 records and 13 sources at /project_docs/ pass at 1440, 768, 390 and 320 px. |
| Browser, isolated sample copy | 43 records and 13 sources at / pass at the same four widths, including the additional course and its project records. |
| Visual review | Sample overview and mobile first lesson inspected: own title, Project 99 · 1 lesson, meaningful project groups, readable layout. Selected images and compact results are in evidence/2026-10-09-p00-skill-standard/. |
| Repository checks | git diff --check passes in CLI, Projects, Projects Docs, Research and Research Docs. Existing unrelated edits preserved. |

The browser suites check page errors, links, images, equations, declared visual/practice controls, navigation, downloads, print action and width containment. These checks validate the local authoring/reader workflow; they do not measure teaching effectiveness.

## Manual instruction review

This is a review of the written routes, not an independent agent evaluation or a learner session.

| Request exercised against the instructions | Expected behavior now specified |
| --- | --- |
| Create the next project tutorial | Inspect the actual roadmap, choose a reachable first action, map maths/prerequisites, author integrated lessons and project evidence records, execute the file sequence, then do four review passes. Adapt the number of lessons to scope. |
| Combine chapters and build lessons | Inventory explanations, code, maths, IDs, sources and results; place the depth at its use; preserve old routes and downloads; remove competing navigation after content preservation checks. |
| Do a second maths/readiness pass | Inspect first-use background, symbols, assumptions, worked calculations and independent checks; fix gaps and map required next-project evidence rather than merely inserting a checklist. |
| Teach interactively | Use the adaptive reference: one reachable question, wait for the learner, then respond to the actual answer. Artifact authoring does not wait for diagnostic answers. |
| Explain one short idea | Give a motivated worked example at the requested depth; do not create an unnecessary course or project scaffold. |

## Continuation and limits

The user can invoke `$developer-learning-tutorials` from either owning umbrella for the P00 teaching approach. This is the workspace skill; the independently scoped installed Dashboard skill is unchanged. Templates remain prompts for real authoring, not a guarantee that generated prose is finished.

Active learning remains **P00-M1 not-started**; Research remains deferred intake. Next learner action is P00 Lesson 1, followed by Lesson 2 and the independent transformer-foundations-lab kickoff. No learner/model repository, training, hosted recovery, measured mastery, remote publication or commit was created by this skill update. Canonical preparation records and all six compact snapshots were refreshed.
