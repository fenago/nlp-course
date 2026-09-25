// main.js: wires the buttons to the steps and posts every result to serve.py.
// You do not need to change this file.
import { $, log, loadNotices, loadQueries, report, predict, keywordSearch, reportResources, showHits } from './lab.js';
import * as step1 from './step1.js';
import * as step2 from './step2.js';
import * as step3 from './step3.js';
import * as own from './own.js';

const round = (v) => Array.from(v, (x) => Math.round(x * 1e6) / 1e6);
const ms = (t0) => Math.round(performance.now() - t0);
let notices = null;
let rowsAtStart = null;
let loadMs = null;

async function withNotices() {
  if (!notices) notices = await loadNotices();
  return notices;
}

function busy(fn) {
  return async (ev) => {
    const b = ev?.target;
    if (b) b.disabled = true;
    try { await fn(); } catch (e) { log(`ERROR: ${e.message}`); console.error(e); }
    finally { if (b) b.disabled = false; await reportResources(); }
  };
}

async function queryVector(q) {
  return (await step1.embed([q]))[0];
}

async function s1Store() {
  const ns = await withNotices();
  let t0 = performance.now();
  await step1.loadModel();
  loadMs = ms(t0);
  log(`embedding model loaded in ${loadMs} ms`);
  t0 = performance.now();
  const vecs = await step1.embed(ns.map((n) => `${n.title}. ${n.text}`));
  const embedMs = ms(t0);
  ns.forEach((n, i) => { n.vector = vecs[i]; });
  const stored = await step1.storeNotices(ns.map((n) => ({ id: n.id, title: n.title, text: n.text, vector: n.vector })));
  log(`embedded ${ns.length} notices in ${embedMs} ms (${vecs[0].length} numbers each); IndexedDB holds ${stored}`);
  window.__s1 = { embedMs, stored, dims: vecs[0].length };
}

async function s1Search() {
  const ns = await withNotices();
  if (!window.__s1) await s1Store();
  const q = $('q').value;
  let t0 = performance.now();
  const qv = await queryVector(q);
  const embedQueryMs = ms(t0);
  t0 = performance.now();
  const hits = await step1.searchIndexedDB(qv, 3);
  const searchMs = ms(t0);
  const kw = keywordSearch(ns, q, 3);
  showHits($('r1'), hits); showHits($('r1k'), kw);
  log(hits.length ? `IndexedDB search: ${hits.map((h) => h.id).join(', ')} in ${searchMs} ms` : 'IndexedDB search returned nothing: searchIndexedDB in step1.js is not written yet');
  const guess = $('p1').value;
  if (guess) await predict('p13_keyword', { guess });
  await predict('p13_keyword', { actual: kw[0]?.id === 'n08' && kw[0].score > 0 ? 'yes' : 'no' });
  const vectors = {};
  for (const n of ns) vectors[n.id] = round(n.vector);
  await report('step1', {
    model: step1.MODEL, count: ns.length, dims: window.__s1.dims, stored: window.__s1.stored,
    timings: { load_ms: loadMs, embed_ms: window.__s1.embedMs, embed_query_ms: embedQueryMs, search_ms: searchMs },
    query: q, query_vector: round(qv), hits, keyword_hits: kw, vectors,
  });
}

async function s2Status() {
  const { rows, info } = await step2.status();
  $('s2-status').textContent = `SQLite ${info.sqlite}, storage ${info.vfs}, FTS5 ${info.fts5 ? 'included' : 'missing'}: ${rows} rows in notices`;
  return { rows, info };
}

async function s2Store() {
  const ns = await withNotices();
  if (!ns[0].vector) await s1Store();
  const t0 = performance.now();
  const n = await step2.storeNotices(ns);
  log(`SQLite now holds ${n} rows (stored in ${ms(t0)} ms)`);
  await s2Status();
}

async function s2Search() {
  const q = $('q').value;
  const qv = await queryVector(q);
  const t0 = performance.now();
  const hits = await step2.searchSQLite(qv, 3);
  const searchMs = ms(t0);
  showHits($('r2'), hits);
  log(hits.length ? `SQLite search: ${hits.map((h) => h.id).join(', ')} in ${searchMs} ms` : 'SQLite search returned nothing: SQL_SEARCH in step2.js is empty, or the table is empty');
  const { rows, info } = await s2Status();
  const guess = $('p2').value;
  if (guess) await predict('p13_reload', { guess });
  if (rowsAtStart > 0) await predict('p13_reload', { actual: String(rowsAtStart) });
  await report('step2', { ...info, rows_at_start: rowsAtStart, rows, sql_search: step2.SQL_SEARCH, query: q, query_vector: round(qv), hits, timings: { search_ms: searchMs } });
}

async function s3Answer() {
  const q = $('q').value;
  const qv = await queryVector(q);
  let hits = await step2.searchSQLite(qv, 2);
  if (!hits.length) hits = await step1.searchIndexedDB(qv, 2);
  const ns = await withNotices();
  const context = hits.map((h) => ns.find((n) => n.id === h.id));
  if (!step3.buildMessages(q, context).length) {
    $('a3').textContent = '(no answer: buildMessages in step3.js is not written yet)';
    log('step 3: buildMessages returned no messages, so the model was not loaded');
    return report('step3', { model: step3.MODEL, question: q, context_ids: hits.map((h) => h.id), messages: [], answer: '', new_tokens: 0 });
  }
  log(`loading ${step3.MODEL} (about 137 MB the first time) and writing an answer`);
  let t0 = performance.now();
  await step3.loadGenerator();
  const load = ms(t0);
  t0 = performance.now();
  const r = await step3.answer(q, context);
  const gen = ms(t0);
  $('a3').textContent = r.answer || '(no answer: buildMessages in step3.js is not written yet)';
  log(`answer: ${r.new_tokens} tokens in ${gen} ms`);
  if ($('p3').value) await predict('p13_answer', { guess: $('p3').value });
  await predict('p13_answer', { actual: r.answer.includes('121') ? 'yes' : 'no' });
  await report('step3', { model: step3.MODEL, question: q, context_ids: hits.map((h) => h.id), ...r,
    timings: { load_ms: load, gen_ms: gen }, tokens_per_s: r.new_tokens ? +(r.new_tokens / (gen / 1000)).toFixed(2) : 0 });
}

async function ownTry() {
  const hits = await own.hybridSearch($('q').value, 3);
  showHits($('r4'), hits);
}

async function ownEval() {
  const qs = await loadQueries();
  const t0 = performance.now();
  const results = [];
  for (const q of qs) results.push({ id: q.id, query: q.query, top: (await own.hybridSearch(q.query, 3)).map((h) => h.id) });
  const total = ms(t0);
  const vectorOnly = [];
  for (const q of qs) vectorOnly.push({ id: q.id, top: (await step2.searchSQLite(await queryVector(q.query), 3)).map((h) => h.id) });
  showHits($('r4'), results.map((r) => ({ id: r.id, title: `${r.top.join(' ')}  ${r.query}` })));
  log(`evaluated ${results.length} questions in ${total} ms`);
  await report('own', { description: own.DESCRIPTION, results, vector_only: vectorOnly, timings: { total_ms: total } });
}

$('b1-store').onclick = busy(s1Store);
$('b1-search').onclick = busy(s1Search);
$('b2-store').onclick = busy(s2Store);
$('b2-search').onclick = busy(s2Search);
$('b3').onclick = busy(s3Answer);
$('b4-try').onclick = busy(ownTry);
$('b4-eval').onclick = busy(ownEval);
$('where').textContent = window.top !== window ? `Served from ${location.host}, shown in a frame.` : `Served from ${location.host}.`;

// On every load, ask SQLite how many rows it already has: the reload test.
(async () => {
  try { rowsAtStart = (await s2Status()).rows; log(`page loaded; SQLite already holds ${rowsAtStart} rows`); }
  catch (e) { log(`SQLite did not open: ${e.message}`); }
  await reportResources();
  // ?auto=1 and ?auto=2 run everything unattended; the lab's own walk uses them.
  const auto = new URLSearchParams(location.search).get('auto');
  const run = async (f) => { try { await f(); } catch (e) { log(`ERROR: ${e.message}`); } };
  if (auto === '1') {
    $('p1').value = 'yes'; $('p2').value = '48'; $('p3').value = 'yes';
    await run(s1Search); await run(s2Store); await run(s2Search);
    await reportResources();
    location.replace(location.pathname + '?auto=2');
  } else if (auto === '2') {
    await run(s2Search); await run(ownEval);
    if (new URLSearchParams(location.search).get('answer') !== 'no') await run(s3Answer);
    await reportResources();
    log('auto run finished');
  }
})();
