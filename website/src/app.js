import './styles.css';
import 'katex/dist/katex.min.css';

const query = document.querySelector('#record-search');
function filterRecords() {
  let count = 0;
  for (const record of document.querySelectorAll('.record-card')) {
    record.hidden = !record.dataset.search.includes(query.value.trim().toLowerCase());
    if (!record.hidden) count++;
  }
  document.querySelector('#record-status').textContent = `${count} ${count === 1 ? 'record' : 'records'}`;
  document.querySelector('#record-empty').hidden = count !== 0;
}
query?.addEventListener('input', filterRecords);
const sourceQuery = document.querySelector('#source-search');
const collection = document.querySelector('#collection-filter');
const kind = document.querySelector('#kind-filter');
function filterSources() {
  let count = 0;
  for (const row of document.querySelectorAll('.source-row')) {
    row.hidden = !(row.dataset.search.includes(sourceQuery.value.trim().toLowerCase()) && (!collection.value || row.dataset.collection === collection.value) && (!kind.value || row.dataset.kind === kind.value));
    if (!row.hidden) count++;
  }
  document.querySelector('#source-status').textContent = `${count} ${count === 1 ? 'source' : 'sources'}`;
  document.querySelector('#source-empty').hidden = count !== 0;
}
for (const control of [sourceQuery, collection, kind]) {
  control?.addEventListener('input', filterSources);
  control?.addEventListener('change', filterSources);
}
document.querySelector('.print-button')?.addEventListener('click', () => window.print());
for (const equation of document.querySelectorAll('.katex-display')) {
  equation.tabIndex = 0;
  equation.setAttribute('role', 'region');
  equation.setAttribute('aria-label', 'Equation; scroll horizontally if needed');
}
const tutorialMenu = document.querySelector('.tutorial-menu');
if (tutorialMenu && window.matchMedia('(max-width: 1100px)').matches) tutorialMenu.open = false;
for(const button of document.querySelectorAll('.practice-reveal')) button.addEventListener('click',()=>{
  const answer=button.parentElement.querySelector('.practice-answer');answer.hidden=!answer.hidden;
  button.setAttribute('aria-expanded',String(!answer.hidden));
  button.textContent = answer.hidden ? 'Reveal worked answer and rubric' : 'Hide worked answer';
});
for(const section of document.querySelectorAll('.tutorial-scenario')) {
  const data=JSON.parse(section.dataset.scenario);
  const update=()=>{
    const values=Object.fromEntries([...section.querySelectorAll('[data-variable]')].map(input=>[input.dataset.variable,input.value]));
    const selected=data.cases.find(item=>Object.entries(values).every(([key,value])=>item.when[key]===value));
    section.querySelector('[data-outcome]').textContent=selected.outcome;
    section.querySelector('[data-narrative]').textContent=selected.narrative;
  };
  section.addEventListener('change',update);update();
}
let printAnswers=[];
window.addEventListener('beforeprint',()=>{printAnswers=[...document.querySelectorAll('.practice-answer')].map(answer=>[answer,answer.hidden]);for(const [answer] of printAnswers)answer.hidden=false;});
window.addEventListener('afterprint',()=>{for(const [answer,hidden] of printAnswers)answer.hidden=hidden;});
document.addEventListener('keydown', event => {
  if (event.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
    const search = query ?? sourceQuery;
    if (search) { event.preventDefault(); search.focus(); }
  }
});
const projectMenu = document.querySelector('.project-menu');
if (projectMenu && window.matchMedia('(max-width: 1100px)').matches) projectMenu.open = false;
const diagrams = [...document.querySelectorAll('.markdown .language-mermaid')];
if (diagrams.length) {
  const { default: mermaid } = await import('mermaid');
  mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'dark', suppressErrorRendering: true });
  for (const [index, code] of diagrams.entries()) {
    const region = document.createElement('div');
    region.className = 'project-diagram';
    region.tabIndex = 0;
    region.setAttribute('role', 'region');
    region.setAttribute('aria-label', 'Diagram; scroll horizontally if needed');
    try {
      const { svg } = await mermaid.render(`project-diagram-${index}`, code.textContent);
      region.innerHTML = svg;
      const diagram = region.querySelector('svg');
      // Preserve the diagram's text size; wide flows scroll inside the page.
      if (diagram?.viewBox.baseVal.width) diagram.style.width = `${diagram.viewBox.baseVal.width}px`;
      code.parentElement.replaceWith(region);
      if (diagram?.viewBox.baseVal.width > region.clientWidth) {
        const hint = document.createElement('p');
        hint.className = 'diagram-hint';
        hint.textContent = 'Scroll sideways to follow the complete diagram.';
        region.before(hint);
      }
    } catch { code.parentElement.setAttribute('aria-label', 'Diagram source'); }
  }
}
document.documentElement.dataset.ready = 'true';

const videoSearch = document.querySelector('#video-search');
videoSearch?.addEventListener('input', () => {
  let count = 0;
  for (const card of document.querySelectorAll('.video-card')) {
    card.hidden = !card.dataset.search.includes(videoSearch.value.trim().toLowerCase());
    if (!card.hidden) count++;
  }
  document.querySelector('#video-status').textContent = `${count} ${count === 1 ? 'video' : 'videos'}`;
  document.querySelector('#video-empty').hidden = count !== 0;
});

for (const video of document.querySelectorAll('.section-video video')) {
  const showProblem = () => { const message = video.parentElement.querySelector('.video-error'); if (message) message.hidden = false; };
  video.addEventListener('error', showProblem);
  video.querySelector('source')?.addEventListener('error', showProblem);
}
