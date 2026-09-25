// db-worker.js: SQLite, compiled to WebAssembly by the SQLite project, running
// in a Web Worker so that it may use the Origin Private File System (OPFS).
// You do not need to change this file.
//
// Why a worker: OPFS offers fast synchronous file access (the kind a database
// engine needs) only inside a worker, never on the page's main thread.
// Why 'opfs-sahpool': of SQLite's two OPFS back ends, this one needs no special
// Cross-Origin headers, so it also works inside the App tab's frame.
import sqlite3InitModule from '/vendor/@sqlite.org/sqlite-wasm/dist/index.mjs';

let db = null;
let info = null;

async function open() {
  const sqlite3 = await sqlite3InitModule();
  const pool = await sqlite3.installOpfsSAHPoolVfs({ name: 'kittiwake-pool' });
  db = new pool.OpfsSAHPoolDb('/kittiwake.sqlite3');
  // SQL has no idea what a vector is, so teach it one function: cosine(a, b)
  // over two BLOBs holding Float32 numbers. Our vectors are normalised, so the
  // dot product is the cosine.
  db.createFunction({
    name: 'cosine', deterministic: true,
    xFunc: (_ctx, a, b) => {
      if (!a || !b) return null;
      const x = new Float32Array(a.slice().buffer), y = new Float32Array(b.slice().buffer);
      let s = 0;
      for (let i = 0; i < x.length; i++) s += x[i] * y[i];
      return s;
    },
  });
  info = {
    sqlite: sqlite3.version.libVersion,
    vfs: 'opfs-sahpool',
    fts5: !!db.selectValue("SELECT sqlite_compileoption_used('ENABLE_FTS5')"),
  };
}

onmessage = async (e) => {
  const { id, sql, bind } = e.data;
  try {
    if (!db) await open();
    const rows = sql ? db.exec({ sql, bind, rowMode: 'object', returnValue: 'resultRows' }) : [];
    postMessage({ id, rows, info });
  } catch (err) {
    postMessage({ id, error: String(err?.message ?? err), info });
  }
};
