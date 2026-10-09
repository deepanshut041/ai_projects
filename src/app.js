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
for(const button of document.querySelectorAll('.practice-reveal')) button.addEventListener('click',()=>{
  const answer=button.parentElement.querySelector('.practice-answer');answer.hidden=!answer.hidden;
  button.setAttribute('aria-expanded',String(!answer.hidden));
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
document.documentElement.dataset.ready = 'true';
