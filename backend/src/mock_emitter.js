// Simulates Team A's ML model output at a fixed interval — produces varied, realistic alerts for dashboard development and testing.

import { buildAlert } from './context_engine.js';
import { validateAlert } from './schema_validator.js';
import { insertAlert } from './database.js';

const FALL_TYPES     = ['slip', 'trip', 'faint'];
const PRE_ACTIVITIES = ['walking', 'standing', 'bending', 'sitting'];
const POST_STATES    = ['unconscious', 'stunned', 'moving', 'unknown'];

function randomItem(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function randomConfidence() {
  return Math.round((Math.random() * 0.42 + 0.58) * 100) / 100;
}

export function startMockEmitter(io, intervalMs) {
  console.log(`[MockEmitter] Started — emitting every ${intervalMs}ms`);

  setInterval(() => {
    const raw = {
      fall_type:    randomItem(FALL_TYPES),
      pre_activity: randomItem(PRE_ACTIVITIES),
      post_state:   randomItem(POST_STATES),
      confidence:   randomConfidence(),
    };

    const alert = buildAlert(raw);
    const { valid, errors } = validateAlert(alert);

    if (!valid) {
      console.error('[MockEmitter] Validation failed — skipping emit:', errors);
      return;
    }

    insertAlert(alert);
    io.emit('fall_alert', alert);

    console.log(
      `[MockEmitter] ${alert.severity.padEnd(8)} | ${alert.fall_type.padEnd(5)} | ${alert.post_state.padEnd(11)} | conf: ${alert.confidence.toFixed(2)}`
    );
  }, intervalMs);
}
