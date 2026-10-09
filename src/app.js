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
document.addEventListener('keydown', event => {
  if (event.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
    const search = query ?? sourceQuery;
    if (search) { event.preventDefault(); search.focus(); }
  }
});
document.documentElement.dataset.ready = 'true';
