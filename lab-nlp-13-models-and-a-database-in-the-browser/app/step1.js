// step1.js: turn every notice into a vector in the page, keep the vectors in
// IndexedDB, and search them.
import './config.js';
import { pipeline } from '/vendor/@huggingface/transformers/dist/transformers.js';
import { dot } from './lab.js';

export const MODEL = 'Xenova/all-MiniLM-L6-v2';
let extractor = null;

// Worked example: load the embedding model once, then embed a batch of texts.
// dtype 'q8' picks onnx/model_quantized.onnx (8-bit weights, 23 MB);
// device 'wasm' runs it on the CPU through WebAssembly.
export async function loadModel() {
  if (!extractor) extractor = await pipeline('feature-extraction', MODEL, { dtype: 'q8', device: 'wasm' });
  return extractor;
}

export async function embed(texts) {
  const model = await loadModel();
  // Mean pooling averages the token vectors into one vector per text;
  // normalize makes each one length 1, so a dot product is a cosine.
  const out = await model(texts, { pooling: 'mean', normalize: true });
  const [n, dims] = out.dims;
  return Array.from({ length: n }, (_, i) => out.data.slice(i * dims, (i + 1) * dims));
}

// IndexedDB, the database built into every browser: no tables and no SQL,
// just object stores that hold JavaScript values under a key.
function request(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export function openDB() {
  const req = indexedDB.open('kittiwake', 1);
  // Runs only when the database is new: create one object store keyed by id.
  req.onupgradeneeded = () => req.result.createObjectStore('notices', { keyPath: 'id' });
  return request(req);
}

export async function storeNotices(rows) {
  const db = await openDB();
  const tx = db.transaction('notices', 'readwrite');
  for (const row of rows) tx.objectStore('notices').put(row); // row.vector is a Float32Array
  await new Promise((resolve, reject) => { tx.oncomplete = resolve; tx.onerror = () => reject(tx.error); });
  return request(db.transaction('notices').objectStore('notices').count());
}

// Your turn. IndexedDB cannot rank anything for you: it can only hand back
// what it stores. Read every row, score each one against the query, and return
// the best k, highest score first, as objects { id, title, score }.
export async function searchIndexedDB(queryVector, k = 3) {
  const db = await openDB();
  const rows = await request(db.transaction('notices').objectStore('notices').getAll());
  // YOUR CODE HERE: score each row with dot(queryVector, row.vector), sort the
  // rows by that score from highest to lowest, and return the first k as
  // { id: row.id, title: row.title, score }.
  return [];
}
