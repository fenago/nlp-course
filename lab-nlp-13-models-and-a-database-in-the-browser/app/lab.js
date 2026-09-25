// lab.js: small helpers the steps share. You do not need to change this file.

export const $ = (id) => document.getElementById(id);

export function log(msg) {
  const line = `${new Date().toLocaleTimeString()}  ${msg}`;
  console.log(line);
  $('log').textContent = line + '\n' + $('log').textContent;
}

export async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store' });
  if (!r.ok) throw new Error(`${url} answered ${r.status}`);
  return r.json();
}

export const loadNotices = () => getJSON('notices.json');
export const loadQueries = () => getJSON('queries.json');

// Post a report to serve.py, which saves it as ~/exercises/out/13_<name>.json.
// This is the only way the checks, which run on the session's machine, can
// know what happened inside your browser.
export async function report(name, data) {
  const r = await fetch(`/report/${name}`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  });
  const j = await r.json();
  if (j.saved) log(`report saved to ${j.saved}`);
  return j;
}

// Record a prediction (guess) or what happened (actual) in out/predictions.json.
export async function predict(key, fields) {
  await fetch('/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, ...fields }),
  });
}

// Both vectors are normalised to length 1, so their dot product IS their
// cosine similarity: 1 for the same direction, 0 for unrelated.
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// The old way: score a notice by how many of the query's words it contains.
const STOP = new Set('a an and are as at be but by can do does for from has have i if in is it its my me no not of on or so that the their them there they this to up was we will with you your'.split(' '));
export const words = (t) => t.toLowerCase().match(/[a-z0-9']+/g)?.filter((w) => !STOP.has(w)) ?? [];
export function keywordSearch(notices, query, k = 3) {
  const q = new Set(words(query));
  return notices
    .map((n) => {
      const have = new Set(words(n.title + ' ' + n.text));
      return { id: n.id, title: n.title, score: [...q].filter((w) => have.has(w)).length };
    })
    .sort((a, b) => b.score - a.score)
    .slice(0, k);
}

// Every file this page has fetched, as the browser itself recorded it, and any
// request the Content-Security-Policy refused.
export const violations = [];
document.addEventListener('securitypolicyviolation', (e) => {
  violations.push({ blocked: e.blockedURI, directive: e.violatedDirective });
  log(`BLOCKED by the page's security policy: ${e.blockedURI} (${e.violatedDirective})`);
});
export async function reportResources() {
  const urls = [...new Set(performance.getEntriesByType('resource').map((e) => e.name.split('?')[0]))];
  let quota = null;
  try { quota = await navigator.storage.estimate(); } catch { /* not available */ }
  return report('resources', {
    origin: location.origin, urls, violations,
    crossOriginIsolated: self.crossOriginIsolated, framed: window.top !== window,
    storage_used: quota?.usage ?? null, userAgent: navigator.userAgent,
  });
}

export function showHits(el, hits) {
  el.innerHTML = '';
  for (const h of hits) {
    const li = document.createElement('li');
    li.textContent = `${h.id}  ${h.title}` + (h.score !== undefined ? `  (${Number(h.score).toFixed(3)})` : '');
    el.appendChild(li);
  }
}
