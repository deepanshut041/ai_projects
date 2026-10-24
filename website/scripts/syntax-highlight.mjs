import hljs from 'highlight.js';
import bash from 'highlight.js/lib/languages/bash';

// Research's command-heavy guides mostly invoke external programs. Keep the
// Bash grammar, adding executable/option tokens for common workspace commands.
hljs.registerLanguage('bash', api => {
  const grammar = bash(api);
  grammar.contains.unshift(
    { scope: 'title.function', match: /^[ \t]*(?:[~\w./-]*\/)?(?:ai|uv|python(?:3(?:\.\d+)?)?|pip3?|pytest|node|npm|npx|git|docker|curl|wget|rg|ffmpeg|ffprobe)(?=[ \t]|$)/m },
    { scope: 'attr', match: /\B--?[A-Za-z][A-Za-z0-9-]*\b/ },
  );
  return grammar;
});

// Explicit fence languages only: prose/logs and reader control blocks stay literal.
export function highlightCode(code, language) {
  const name = language.trim().toLowerCase();
  if (!name || ['mermaid', 'video', 'text', 'txt', 'plaintext'].includes(name) || !hljs.getLanguage(name)) return '';
  return hljs.highlight(code, { language: name, ignoreIllegals: true }).value;
}
