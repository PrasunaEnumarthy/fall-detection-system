// Context card — shows the most recent alert in full detail with a brief flash effect on each new event.

import React, { useState, useEffect, useRef } from 'react';
import socket from '../socket.js';

const SEVERITY_COLORS = {
  CRITICAL: { color: 'var(--critical)', bg: 'var(--critical-bg)', border: 'var(--critical-border)' },
  HIGH:     { color: 'var(--high)',     bg: 'var(--high-bg)',     border: 'var(--high-border)'     },
  MEDIUM:   { color: 'var(--medium)',   bg: 'var(--medium-bg)',   border: 'var(--medium-border)'   },
  LOW:      { color: 'var(--low)',      bg: 'var(--low-bg)',      border: 'var(--low-border)'      },
};

function formatDateTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleString('en-US', {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false,
  });
}

export default function ContextCard() {
  const [latest, setLatest]     = useState(null);
  const [flashing, setFlashing] = useState(false);
  const flashTimer              = useRef(null);

  useEffect(() => {
    function handleAlert(alert) {
      setLatest(alert);
      setFlashing(true);
      clearTimeout(flashTimer.current);
      flashTimer.current = setTimeout(() => setFlashing(false), 500);
    }

    socket.on('fall_alert', handleAlert);
    return () => {
      socket.off('fall_alert', handleAlert);
      clearTimeout(flashTimer.current);
    };
  }, []);

  if (!latest) {
    return (
      <div
        style={{
          height:         '100%',
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'center',
          fontFamily:     'var(--font-mono)',
          fontSize:       '12px',
          color:          'var(--text-dim)',
        }}
      >
        Awaiting first event...
      </div>
    );
  }

  const sc = SEVERITY_COLORS[latest.severity] ?? SEVERITY_COLORS.HIGH;

  const dataRows = [
    { label: 'Fall Type',        value: latest.fall_type    },
    { label: 'Pre-Activity',     value: latest.pre_activity },
    { label: 'Post-Fall State',  value: latest.post_state   },
    { label: 'Model Confidence', value: `${Math.round(latest.confidence * 100)}%` },
    { label: 'Timestamp',        value: formatDateTime(latest.timestamp) },
  ];

  return (
    <div
      style={{
        display:         'flex',
        flexDirection:   'column',
        height:          '100%',
        overflow:        'hidden',
        transition:      'background 0.3s ease',
        background:      flashing ? sc.bg : 'transparent',
        borderRadius:    '6px',
      }}
    >
      {/* Panel label */}
      <span
        style={{
          fontFamily:    'var(--font-mono)',
          fontSize:      '11px',
          fontWeight:    600,
          color:         'var(--text-muted)',
          textTransform: 'uppercase',
          letterSpacing: '0.08em',
          marginBottom:  '14px',
          flexShrink:    0,
        }}
      >
        Latest Event Context
      </span>

      {/* Severity block */}
      <div
        style={{
          background:   sc.bg,
          border:       `1px solid ${sc.border}`,
          borderRadius: '6px',
          padding:      '16px',
          marginBottom: '16px',
          flexShrink:   0,
        }}
      >
        <div
          style={{
            fontFamily: 'var(--font-display)',
            fontSize:   '32px',
            color:      sc.color,
            marginBottom: '8px',
          }}
        >
          {latest.severity}
        </div>
        <p
          style={{
            fontSize:   '14px',
            color:      'var(--text)',
            lineHeight: 1.6,
          }}
        >
          {latest.message}
        </p>
      </div>

      {/* Data rows */}
      <div style={{ overflowY: 'auto', flexGrow: 1 }}>
        {dataRows.map(({ label, value }, i) => (
          <div
            key={label}
            style={{
              display:       'flex',
              justifyContent:'space-between',
              alignItems:    'center',
              padding:       '9px 0',
              borderBottom:  i < dataRows.length - 1 ? '1px solid var(--border-light)' : 'none',
            }}
          >
            <span
              style={{
                fontSize:   '12px',
                color:      'var(--text-muted)',
              }}
            >
              {label}
            </span>
            <span
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize:   '12px',
                color:      'var(--text)',
              }}
            >
              {value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
