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
      id                     INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp              TEXT    NOT NULL,
      fall_type              TEXT    NOT NULL,
      pre_activity           TEXT    NOT NULL,
      post_state             TEXT    NOT NULL,
      location               TEXT    DEFAULT 'location_unknown',
      severity               TEXT    NOT NULL,
      message                TEXT    NOT NULL,
      confidence             REAL    NOT NULL,
      confirmation_window_ms INTEGER,
      created_at             TEXT    DEFAULT (datetime('now'))
    )
  `);

  // Migration for databases created before confirmation_window_ms was added.
  // SQLite's ALTER TABLE ADD COLUMN has no IF NOT EXISTS, so we swallow the
  // "duplicate column" error that fires when the column already exists.
  try {
    db.run(`ALTER TABLE alerts ADD COLUMN confirmation_window_ms INTEGER`);
  } catch (_) { /* column already present — nothing to do */ }

  try {
    db.run(`ALTER TABLE alerts ADD COLUMN location TEXT DEFAULT 'location_unknown'`);
  } catch (_) { /* column already present - nothing to do */ }

  persist();
}

export function insertAlert(alert) {
  const stmt = db.prepare(`
    INSERT INTO alerts (timestamp, fall_type, pre_activity, post_state, location, severity, message, confidence, confirmation_window_ms)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  stmt.run([
    alert.timestamp,
    alert.fall_type,
    alert.pre_activity,
    alert.post_state,
    alert.location ?? 'location_unknown',
    alert.severity,
    alert.message,
    alert.confidence,
    alert.confirmation_window_ms ?? null,   // null for alerts without the confirmation feature
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
