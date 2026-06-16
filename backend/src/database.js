// Pure-JS SQLite interface using sql.js — handles table creation, inserts, queries, and on-disk persistence without native compilation.

import initSqlJs from 'sql.js';
import { readFileSync, writeFileSync, existsSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = dirname(__filename);
const DB_PATH    = join(__dirname, '..', 'data', 'alerts.db');

let db = null;

function persist() {
  const data = db.export();
  writeFileSync(DB_PATH, Buffer.from(data));
}

export async function initDatabase() {
  const SQL = await initSqlJs();

  if (existsSync(DB_PATH)) {
    const buf = readFileSync(DB_PATH);
    db = new SQL.Database(buf);
  } else {
    db = new SQL.Database();
  }

  db.run(`
    CREATE TABLE IF NOT EXISTS alerts (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp   TEXT    NOT NULL,
      fall_type   TEXT    NOT NULL,
      pre_activity TEXT   NOT NULL,
      post_state  TEXT    NOT NULL,
      severity    TEXT    NOT NULL,
      message     TEXT    NOT NULL,
      confidence  REAL    NOT NULL,
      created_at  TEXT    DEFAULT (datetime('now'))
    )
  `);
  persist();
}

export function insertAlert(alert) {
  const stmt = db.prepare(`
    INSERT INTO alerts (timestamp, fall_type, pre_activity, post_state, severity, message, confidence)
    VALUES (?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run([
    alert.timestamp,
    alert.fall_type,
    alert.pre_activity,
    alert.post_state,
    alert.severity,
    alert.message,
    alert.confidence,
  ]);
  stmt.free();
  persist();
}

export function getRecentAlerts(limit = 100) {
  const stmt = db.prepare('SELECT * FROM alerts ORDER BY id DESC LIMIT ?');
  stmt.bind([limit]);
  const rows = [];
  while (stmt.step()) {
    rows.push(stmt.getAsObject());
  }
  stmt.free();
  return rows;
}

export function clearAlerts() {
  db.run('DELETE FROM alerts');
  persist();
}
