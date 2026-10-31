import { escape as e } from './content.mjs';
import { heading, recordCard } from './templates.mjs';
import { collections } from './projects.mjs';

const retainedCollections = {
  P00: 'transformer-labs', P01: 'language-models',
  'bike-demand-planner': 'foundations',
  'attention-is-all-you-need': 'transformer-labs',
  'document-assistant': 'quantization-lab',
};

// Prefer real project ownership. Retained IDs also make frozen tutorials
// browsable in a standalone Docs checkout without sibling project repositories.
export function tutorialCollectionLibrary(records, projectGroups, base) {
  const groups = collections.map(group => ({ id: group.id, title: group.title, description: group.description,
    url: `${base}tutorials/${group.id}/`, projectDocsURL: projectGroups.find(projectGroup => projectGroup.id === group.id)?.url ?? null, courses: [] }));
  for (const record of records.filter(record => record.tutorial)) {
    const id = record.tutorial.project_id;
    const owner = projectGroups.find(group => group.id === id || group.projects.some(project => project.id === id))
      ?? collections.find(group => group.id === id || group.id === retainedCollections[id]);
    if (!owner) throw new Error(`Assign tutorial project ${id} to a registered collection before building.`);
    const group = groups.find(group => group.id === owner.id);
    record.tutorial.collection = { id: group.id, title: group.title, url: group.url };
    record.search += ' ' + group.title.toLowerCase();
    if (record.tutorial.kind === 'overview') group.courses.push(record);
  }
  return { groups };
}

export function tutorialCollectionCards(groups) {
  return `<div class="collection-grid">${groups.map(group => `<a class="collection-card" href="${group.url}"><span class="eyebrow">${group.courses.length} ${group.courses.length === 1 ? 'tutorial' : 'tutorials'}</span><h2>${e(group.title)} <span aria-hidden="true">↗</span></h2><p>${e(group.description)}</p></a>`).join('')}</div>`;
}

export function tutorialCollectionsHome(library) {
  return `${heading('Tutorials', 'Choose a collection, then follow a tutorial through its lessons, worked examples and checks.')}${tutorialCollectionCards(library.groups)}`;
}

export function tutorialCollectionPage(group, base) {
  const courses = group.courses.filter(record => record.tutorial.courseRole !== 'reference');
  const references = group.courses.filter(record => record.tutorial.courseRole === 'reference');
  return `<nav class="breadcrumbs" aria-label="Breadcrumb"><a href="${base}tutorials/">Tutorials</a><span>/</span><span>${e(group.title)}</span></nav>${heading(group.title, group.description, 'Tutorials')}<p class="section-description">${courses.length} ${courses.length === 1 ? 'tutorial' : 'tutorials'} in this collection.${group.projectDocsURL ? ` Project goals, setup and implementation records are in <a href="${group.projectDocsURL}">Project Docs →</a>` : ''}</p><div class="record-grid tutorial-courses">${courses.map(recordCard).join('')}</div>${references.length ? `<section class="tutorial-references"><h2>Additional exercises</h2><div class="record-grid">${references.map(recordCard).join('')}</div></section>` : ''}${group.courses.length ? '' : '<p class="empty">No tutorials have been added to this collection yet.</p>'}`;
}

export function tutorialsSidebar(library, active = '') {
  return library.groups.map(group => `<a class="collection-nav" href="${group.url}"${active === `tutorials/${group.id}` ? ' aria-current="page"' : ''}><span>${e(group.title)}</span><small>${group.courses.length}</small></a>`).join('');
}
