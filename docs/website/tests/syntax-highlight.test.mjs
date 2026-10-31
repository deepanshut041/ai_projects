import test from 'node:test';
import assert from 'node:assert/strict';
import { renderer, escape } from '../scripts/content.mjs';

const samples = [
  ['python', 'def attend(q):\n    # Keep permitted keys\n    return "token" if q > 0 else None\n'],
  ['py', 'import torch\nresult = torch.ones(2)\n'],
  ['cpp', '#include <vector>\nint main() { return 42; }\n'],
  ['js', 'const result = {name: "attention", count: 2};\n'],
  ['ts', 'const count: number = 2;\n'],
  ['json', '{"model": "small", "steps": 42, "enabled": true}\n'],
  ['bash', '# Local check\nexport MODE="cpu"\npython -m train\n'],
  ['sh', 'echo "ready"\n'],
  ['sql', 'SELECT name FROM runs WHERE score > 0;\n'],
  ['yaml', 'model: small\nsteps: 42\n'],
  ['toml', '[model]\nname = "small"\n'],
  ['rust', 'fn main() { let count = 42; }\n'],
  ['html', '<p class="example">Visible source</p>\n'],
  ['css', '.example { color: red; }\n'],
];
test('declared fence languages and aliases receive token markup without changing code', () => {
  const {md} = renderer(new Set(), 'guides/example.md', '/');
  for (const [language, source] of samples) {
    const html = md.render('```' + language + '\n' + source + '```\n');
    assert.match(html, /<span class="hljs-/, language);
    const literal = html.replace(/<\/?span\b[^>]*>/g, '');
    assert.ok(literal.includes('>' + escape(source) + '</code>'), language + ': source text changed');
    assert.match(html, /<pre tabindex="0" role="region"/);
  }
});
test('plain, missing, unknown and control languages remain escaped literal text', () => {
  const {md} = renderer(new Set(), 'guides/example.md', '/');
  const source = '<script>alert("example")</script>\n';
  for (const language of ['', 'text', 'txt', 'plaintext', 'unknown-example', 'mermaid', 'video']) {
    const html = md.render('```' + language + '\n' + source + '```\n');
    assert.doesNotMatch(html, /hljs-|<script>/, language);
    assert.ok(html.includes(escape(source)), language);
  }
});
test('workspace shell invocations and flags are visible without coloring quoted options', () => {
  const {md} = renderer(new Set(), 'guides/example.md', '/');
  const html = md.render('```bash\n./ai research study run --seed 7\nuv run --frozen python train.py\necho "--literal"\n```\n');
  assert.match(html, /class="hljs-title function_">\.\/ai<\/span>/);
  assert.match(html, /class="hljs-attr">--seed<\/span>/);
  assert.match(html, /class="hljs-title function_">uv<\/span>/);
  assert.match(html, /class="hljs-string">&quot;--literal&quot;<\/span>/);
});
test('highlighted markup source stays inert and inline code remains unchanged', () => {
  const {md} = renderer(new Set(), 'guides/example.md', '/');
  const html = md.render('```html\n<img src=x onerror="alert(1)">\n```\n');
  assert.match(html, /hljs-name/);
  assert.doesNotMatch(html, /<img\b|<script\b/);
  assert.equal(md.renderInline('`return "<tag>"`'), '<code>return &quot;&lt;tag&gt;&quot;</code>');
});
