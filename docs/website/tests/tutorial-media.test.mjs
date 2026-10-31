import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { localTutorialVideos } from '../scripts/tutorials.mjs';
import { renderer } from '../scripts/content.mjs';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
test('tutorial movies require an explicit unchanged finished file; exports keep readable companions', async () => {
  const workspace = await fs.mkdtemp(path.join(os.tmpdir(), 'tutorial-media-'));
  try {
    const directory = path.join(workspace, 'course');
    const movie = path.join(workspace, '.ai/videos/projects/example/out/video.mp4');
    await fs.mkdir(directory);
    await fs.mkdir(path.dirname(movie), { recursive: true });
    await fs.writeFile(movie, 'finished movie fixture');
    const item = { src: 'media/example/video.mp4', package: 'example', file: 'out/video.mp4', sha256: hash(await fs.readFile(movie)) };
    const manifest = { schema_version: 1, videos: [item] };
    const save = () => fs.writeFile(path.join(directory, 'local-videos.json'), JSON.stringify(manifest));
    await save();
    const frozen = { files: { 'local-videos.json': 'fixture' } };
    const local = await localTutorialVideos(directory, frozen, workspace, true);
    assert.equal(Object.values(local.files)[0], movie);
    assert.match(local.links[item.src], /^local-files\/[a-f0-9]{24}\/video\.mp4$/);
    const video = { title: 'A worked hour', duration: 45, src: item.src, poster: 'media/example/poster.jpg', captions: 'media/example/captions.vtt', transcript: 'media/example/transcript.md' };
    const metadata = { links: { ...local.links, [video.poster]: video.poster, [video.captions]: video.captions, [video.transcript]: video.transcript }, localVideos: local.sources };
    const block = '```video\n' + JSON.stringify(video) + '\n```';
    assert.match(renderer(new Set(), '', '/nested/', metadata).md.render(block), /<video controls playsinline/);
    const publicMap = await localTutorialVideos(directory, frozen, workspace, false);
    assert.deepEqual(publicMap.files, {});
    const exported = renderer(new Set(), '', '/nested/', { ...metadata, links: { ...metadata.links, ...publicMap.links } }).md.render(block);
    assert.doesNotMatch(exported, /<video|local-files\/|\.ai\//);
    assert.match(exported, /Read the complete transcript/);
    assert.match(exported, /href="\/nested\/media\/example\/transcript.md"/);
    await fs.writeFile(movie, 'different movie bytes');
    await assert.rejects(localTutorialVideos(directory, frozen, workspace, true), /missing or changed/);
    item.file = '../video.mp4'; await save();
    await assert.rejects(localTutorialVideos(directory, frozen, workspace, false), /Unsafe/);
    item.file = 'out/video.mp4'; manifest.videos.push(item); await save();
    await assert.rejects(localTutorialVideos(directory, frozen, workspace, false), /Unsafe/);
  } finally { await fs.rm(workspace, { recursive: true, force: true }); }
});
