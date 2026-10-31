import fs from 'node:fs/promises';
import path from 'node:path';

// All canonical Projects documentation is owned by this docs repository.
export async function workspaceRecords(siteRoot, companionUrl, tutorialPages = {}) {
  const owner = path.resolve(siteRoot);
  const definitions = [
    ['guide/getting-started.md', 'start-here', 'Start your first project', 'P00 kickoff, workspace setup, commands and repository map in one guide.'],
    ['guide/COLAB.md', 'colab', 'Colab training and recovery', 'Free-first early-model training, persistent checkpoints and recovery after runtime termination.'],
    ['guide/roadmap.md', 'roadmap', 'Projects learning roadmap', 'The complete known-model learning path, prerequisites and completion gates.'],
    ['guide/progress.md', 'progress', 'Projects progress', 'Evidence and milestone states from the canonical learning ledger.'],
    ['guide/resume.md', 'resume', 'Resume Projects', 'The current project and concrete next action.'],
    ['guide/decisions.md', 'decisions', 'Projects decisions', 'Durable learning and ownership choices.'],
    ['guide/attention-labs-route.md', 'attention-labs-route', 'Attention Labs reading route', 'The ordered attention tutorial continuation, grouped by mechanism and category.'],
    ['implementation/2026-10/project-zero-foundations-and-small-gpt-roadmap/02-implementation-plan.md', 'project-zero-plan', 'Project Zero and small GPT plan', 'Ground-up tokenizer, attention and complete Transformer milestones before a decoder-only text model.'],
    ['implementation/2026-10/llm-roadmap-review-and-gpt-1-starting-plan/02-implementation-plan.md', 'gpt-1-starting-plan', 'GPT-1 starting plan (historical reference)', 'Preserved earlier plan; the live P00-first roadmap supersedes this exact GPT-1 starting scope.'],
  ];
  const sources = [];
  for (const [file, id, title, description] of definitions) {
    const absolute = path.join(owner, file);
    try {
      const body = await fs.readFile(absolute, 'utf8');
      const updated = body.match(/^Updated:\s*(\d{4}-\d{2}-\d{2})/m)?.[1] ?? (await fs.stat(absolute)).mtime.toISOString().slice(0, 10);
      sources.push({ absolute, id, title, description, body, updated });
    }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
  if (!sources.some(source => source.id === 'roadmap')) return [];
  for (const id of ['roadmap', 'progress', 'resume', 'decisions']) {
    if (!sources.some(source => source.id === id)) throw new Error(`Missing canonical workspace record: ${id}.md`);
  }
  const targets = new Map(sources.map(source => [source.absolute, `guides/${source.id}.md`]));
  for (const [source, file] of Object.entries(tutorialPages)) {
    // A canonical lesson.md link still opens the project home; individual
    // lesson sources link directly to their integrated reading page.
    if (!source.endsWith('/lesson.md')) targets.set(path.join(owner, source), file);
  }
  // Portable aliases keep links from the unchanged tutorials and old bookmarks valid.
  targets.set(path.join(owner, 'roadmap.md'), 'guides/roadmap.md');
  targets.set(path.join(owner, 'START_HERE.md'), 'guides/start-here.md');
  return sources.map(({ absolute, id, title, description, body, updated }) => {
    body = body.replace(/^# [^\n]*\n(?:\n)?/, ''); // recordPage supplies the page's H1.
    if (id === 'gpt-1-starting-plan') {
      body = '**Historical reference:** the [current roadmap](roadmap.md#project-0-tokenization-and-transformer-foundations) starts with P00, then a small GPT-style model. This preserved plan is optional GPT-1 study and does not define the current checkpoint.\n\n' + body;
    }
    body = body.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (original, label, href) => {
      if (/^(?:https?:|mailto:|#)/i.test(href)) return original;
      const [file, fragment] = href.split('#');
      const target = path.resolve(path.dirname(absolute), file);
      if (targets.has(target)) return `[${label}](${path.posix.relative('guides', targets.get(target))}${fragment ? '#' + fragment : ''})`;
      const lesson = path.relative(owner, target).replaceAll(path.sep, '/').match(/^tutorials\/projects\/(p\d{2})\/([a-z][a-z0-9-]*)\/([a-z][a-z0-9-]*)\/lesson\.md$/);
      if (lesson) return `[${label}](../tutorials/${lesson[1]}-${lesson[2]}-${lesson[3]}.md${fragment ? '#' + fragment : ''})`;
      if (companionUrl && target === path.resolve(owner, '../../ai_research/roadmap.md')) return `[${label}](${companionUrl})`;
      return `${label} (local workspace file: \`${href}\`)`;
    });
    const raw = `---\ntitle: ${JSON.stringify(title)}\ndescription: ${JSON.stringify(description)}\nupdated: ${JSON.stringify(updated)}\nstatus: "canonical workspace record"\ntags: ["guides", "roadmap"]\n---\n\n${body}`;
    return { file: `guides/${id}.md`, raw };
  });
}
