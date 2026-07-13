// Renders a single fall alert as a styled card with severity coloring, tag pills, and a low-confidence warning badge.

import React from 'react';

const SEVERITY_COLORS = {
  CRITICAL: { color: 'var(--critical)', bg: 'var(--critical-bg)', border: 'var(--critical-border)' },
  HIGH:     { color: 'var(--high)',     bg: 'var(--high-bg)',     border: 'var(--high-border)'     },
  MEDIUM:   { color: 'var(--medium)',   bg: 'var(--medium-bg)',   border: 'var(--medium-border)'   },
  LOW:      { color: 'var(--low)',      bg: 'var(--low-bg)',      border: 'var(--low-border)'      },
};

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

function formatLocation(location) {
  return location === 'location_unknown' || !location ? 'Unknown' : location;
}

export default function AlertCard({ alert, isNew }) {
  const sc = SEVERITY_COLORS[alert.severity] ?? SEVERITY_COLORS.HIGH;
  const isLowConf = alert.confidence < 0.70;

  return (
    <div
      style={{
        background:       sc.bg,
        border:           `1px solid ${sc.border}`,
        borderLeft:       `3px solid ${sc.color}`,
        borderRadius:     '6px',
        padding:          '12px 14px',
        animation:        isNew ? 'slideIn 0.25s ease-out both' : 'none',
      }}
    >
      {/* Top row: badges + timestamp */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '7px', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
          <span
            style={{
              fontFamily:  'var(--font-mono)',
              fontSize:    '10px',
              fontWeight:  600,
              color:       sc.color,
              background:  'transparent',
              border:      `1px solid ${sc.color}`,
              borderRadius:'20px',
              padding:     '2px 8px',
              letterSpacing: '0.04em',
            }}
          >
            {alert.severity}
          </span>
          {isLowConf && (
            <span
              style={{
                fontFamily:  'var(--font-mono)',
                fontSize:    '10px',
                fontWeight:  500,
                color:       'var(--medium)',
                border:      '1px solid var(--medium-border)',
                borderRadius:'20px',
                padding:     '2px 8px',
              }}
            >
              Low Confidence
            </span>
          )}
        </div>
        <span
          style={{
            fontFamily: 'var(--font-mono)',
            fontSize:   '11px',
            color:      'var(--text-muted)',
            flexShrink: 0,
          }}
        >
          {formatTime(alert.timestamp)}
        </span>
      </div>

      {/* Message */}
      <p
        style={{
          fontSize:   '13px',
          color:      'var(--text)',
          lineHeight: 1.5,
          marginBottom: '10px',
        }}
      >
        {alert.message}
      </p>

      {/* Tag pills */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px' }}>
        {[
          `📍 ${formatLocation(alert.location)}`,
          alert.fall_type,
          alert.pre_activity,
          alert.post_state,
          `conf: ${Math.round(alert.confidence * 100)}%`,
        ].map((tag) => (
          <span
            key={tag}
            style={{
              fontFamily:   'var(--font-mono)',
              fontSize:     '10px',
              color:        'var(--text-muted)',
              background:   'var(--surface-2)',
              border:       '1px solid var(--border)',
              borderRadius: '4px',
              padding:      '2px 7px',
            }}
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}
