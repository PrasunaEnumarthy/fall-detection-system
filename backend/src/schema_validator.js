// Validates every incoming alert object against the shared Team A↔B contract schema before any processing or persistence.

const VALID_FALL_TYPES = ['slip', 'trip', 'faint'];
const VALID_PRE_ACTIVITIES = ['walking', 'standing', 'bending', 'sitting'];
const VALID_POST_STATES = ['unconscious', 'stunned', 'moving', 'unknown'];
const VALID_SEVERITIES = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];

export function validateAlert(alert) {
  const errors = [];

  if (!alert || typeof alert !== 'object') {
    return { valid: false, errors: ['Alert must be a non-null object'] };
  }

  // timestamp — must parse as valid date
  if (!alert.timestamp) {
    errors.push('timestamp is required');
  } else if (isNaN(Date.parse(alert.timestamp))) {
    errors.push(`timestamp "${alert.timestamp}" is not a valid ISO date string`);
  }

  // fall_type
  if (!alert.fall_type) {
    errors.push('fall_type is required');
  } else if (!VALID_FALL_TYPES.includes(alert.fall_type)) {
    errors.push(`fall_type "${alert.fall_type}" must be one of: ${VALID_FALL_TYPES.join(', ')}`);
  }

  // pre_activity
  if (!alert.pre_activity) {
    errors.push('pre_activity is required');
  } else if (!VALID_PRE_ACTIVITIES.includes(alert.pre_activity)) {
    errors.push(`pre_activity "${alert.pre_activity}" must be one of: ${VALID_PRE_ACTIVITIES.join(', ')}`);
  }

  // post_state
  if (!alert.post_state) {
    errors.push('post_state is required');
  } else if (!VALID_POST_STATES.includes(alert.post_state)) {
    errors.push(`post_state "${alert.post_state}" must be one of: ${VALID_POST_STATES.join(', ')}`);
  }

  // severity
  if (!alert.severity) {
    errors.push('severity is required');
  } else if (!VALID_SEVERITIES.includes(alert.severity)) {
    errors.push(`severity "${alert.severity}" must be one of: ${VALID_SEVERITIES.join(', ')}`);
  }

  // message
  if (!alert.message) {
    errors.push('message is required');
  } else if (typeof alert.message !== 'string') {
    errors.push('message must be a string');
  } else if (alert.message.length <= 5) {
    errors.push('message must be longer than 5 characters');
  }

  // confidence
  if (alert.confidence === undefined || alert.confidence === null) {
    errors.push('confidence is required');
  } else if (typeof alert.confidence !== 'number') {
    errors.push('confidence must be a number');
  } else if (alert.confidence < 0 || alert.confidence > 1) {
    errors.push(`confidence ${alert.confidence} must be between 0.0 and 1.0`);
  }

  return { valid: errors.length === 0, errors };
}
