// own.js: Part 3, on your own. Nothing here is written for you.
//
// The brief: some of Kittiwake's questions are about meaning ("my charger
// gets really hot") and some are about exact strings no embedding model
// understands ("E42", "18001"). Write hybridSearch so that it answers both
// kinds, using the SQLite database from step 2. The Evaluate button runs your
// function on the 16 questions in queries.json and reports the top notices for
// each; the checkpoint scores them against answers it holds.
//
// You may use anything the earlier steps export, for example:
//   embed([text])                      from step1.js  -> [Float32Array]
//   sql(text, bind)                    from step2.js  -> { rows, info }
//   searchSQLite(vector, k)            from step2.js  -> [{ id, title, score }]
// and SQLite's full-text search, FTS5, which this build of SQLite includes.

// One or two sentences saying what your method does. The sign-off shows it.
export const DESCRIPTION = '';

// Return the best k notices for the query, best first: [{ id, title, score }].
export async function hybridSearch(query, k = 3) {
  throw new Error('hybridSearch is not written yet: this is Part 3, on your own');
}
