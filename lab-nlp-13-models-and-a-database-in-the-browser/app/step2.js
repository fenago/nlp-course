// step2.js: the same notices and vectors in a real SQL database, SQLite,
// stored in the browser's Origin Private File System so it survives a reload.

const worker = new Worker('db-worker.js', { type: 'module' });
let next = 0;
const waiting = new Map();
worker.onmessage = (e) => {
  const { id, rows, error, info } = e.data;
  const { resolve, reject } = waiting.get(id);
  waiting.delete(id);
  error ? reject(new Error(error)) : resolve({ rows, info });
};

// Run SQL in the worker. bind fills the ? placeholders, in order.
export function sql(text, bind = undefined) {
  return new Promise((resolve, reject) => {
    const id = ++next;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, sql: text, bind });
  });
}

// A Float32Array as the bytes SQLite stores in a BLOB column.
export const toBlob = (vector) => new Uint8Array(Float32Array.from(vector).buffer);

export const SQL_CREATE = `
  CREATE TABLE IF NOT EXISTS notices (
    id     TEXT PRIMARY KEY,
    title  TEXT NOT NULL,
    text   TEXT NOT NULL,
    vector BLOB NOT NULL      -- 384 Float32 numbers, 1,536 bytes
  )`;

export const SQL_INSERT = `INSERT OR REPLACE INTO notices (id, title, text, vector) VALUES (?, ?, ?, ?)`;

export const SQL_COUNT = `SELECT count(*) AS n FROM notices`;

// Your turn. Write ONE SELECT that ranks the notices by meaning, in SQL.
// It gets two ? placeholders, filled in this order:
//   1. the query's vector, as a BLOB
//   2. k, how many rows to return
// Return the columns id, title, and cosine(vector, ?) AS score,
// best first, and only k rows.
// YOUR CODE HERE
export const SQL_SEARCH = ``;

export async function status() {
  const { rows, info } = await sql(`SELECT count(*) AS n FROM sqlite_master WHERE type = 'table' AND name = 'notices'`);
  const n = rows[0].n ? (await sql(SQL_COUNT)).rows[0].n : 0;
  return { rows: n, info };
}

export async function storeNotices(notices) {
  await sql(SQL_CREATE);
  for (const n of notices) await sql(SQL_INSERT, [n.id, n.title, n.text, toBlob(n.vector)]);
  return (await sql(SQL_COUNT)).rows[0].n;
}

export async function searchSQLite(queryVector, k = 3) {
  if (!SQL_SEARCH.trim()) return [];
  return (await sql(SQL_SEARCH, [toBlob(queryVector), k])).rows;
}
