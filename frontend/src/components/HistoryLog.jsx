// Persistent history table — shows all alerts this session with sticky headers, alternating rows, and a clear button.

import React, { useState, useEffect } from 'react';
import socket from '../socket.js';

const SEVERITY_COLORS = {
  CRITICAL: 'var(--critical)',
  HIGH:     'var(--high)',
  MEDIUM:   'var(--medium)',
  LOW:      'var(--low)',
};

function formatTime(isoString) {
  const d = new Date(isoString);
  return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
}

const COL_HEADERS = ['Time', 'Severity', 'Fall Type', 'Pre-Activity', 'Post-State', 'Confidence'];

export default function HistoryLog({ initialHistory }) {
  const [history, setHistory] = useState([]);

  // Load initial history from API on mount
  useEffect(() => {
    if (Array.isArray(initialHistory) && initialHistory.length > 0) {
      setHistory(initialHistory);
    }
  }, [initialHistory]);

  useEffect(() => {
    function handleAlert(alert) {
      setHistory((prev) => [alert, ...prev]);
    }

    function handleCleared() {
      setHistory([]);
    }

    socket.on('fall_alert', handleAlert);
    socket.on('alerts_cleared', handleCleared);

    return () => {
      socket.off('fall_alert', handleAlert);
      socket.off('alerts_cleared', handleCleared);
    };
  }, []);

  async function handleClear() {
    try {
      await fetch('/api/alerts', { method: 'DELETE' });
      setHistory([]);
    } catch (err) {
      console.error('Failed to clear alerts:', err);
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Panel header */}
      <div
        style={{
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'space-between',
          marginBottom:   '12px',
          flexShrink:     0,
        }}
      >
        <span
          style={{
            fontFamily:    'var(--font-mono)',
            fontSize:      '11px',
            fontWeight:    600,
            color:         'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          Session History
          <span
            style={{
              marginLeft: '8px',
              fontWeight: 400,
              color:      'var(--text-dim)',
            }}
          >
            ({history.length} events)
          </span>
        </span>
        <button
          onClick={handleClear}
          style={{
            fontFamily:   'var(--font-mono)',
            fontSize:     '11px',
            color:        'var(--text-muted)',
            background:   'transparent',
            border:       '1px solid var(--border)',
            borderRadius: '4px',
            padding:      '4px 12px',
            cursor:       'pointer',
          }}
          onMouseEnter={(e) => { e.target.style.borderColor = 'var(--critical)'; e.target.style.color = 'var(--critical)'; }}
          onMouseLeave={(e) => { e.target.style.borderColor = 'var(--border)'; e.target.style.color = 'var(--text-muted)'; }}
        >
          Clear All
        </button>
      </div>

      {/* Table */}
      <div style={{ overflowY: 'auto', flexGrow: 1 }}>
        {history.length === 0 ? (
          <div
            style={{
              padding:    '40px 0',
              textAlign:  'center',
              fontFamily: 'var(--font-mono)',
              fontSize:   '12px',
              color:      'var(--text-dim)',
            }}
          >
            No events recorded this session
          </div>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr
                style={{
                  position: 'sticky',
                  top:      0,
                  background: 'var(--surface)',
                  zIndex:   1,
                }}
              >
                {COL_HEADERS.map((h) => (
                  <th
                    key={h}
                    style={{
                      fontFamily:    'var(--font-mono)',
                      fontSize:      '10px',
                      fontWeight:    600,
                      color:         'var(--text-muted)',
                      textTransform: 'uppercase',
                      letterSpacing: '0.07em',
                      textAlign:     'left',
                      padding:       '6px 10px',
                      borderBottom:  '1px solid var(--border)',
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {history.map((alert, i) => (
                <tr
                  key={`${alert.timestamp}-${i}`}
                  style={{
                    background: i % 2 === 1 ? 'rgba(0,0,0,0.02)' : 'transparent',
                  }}
                >
                  <td style={tdStyle}>{formatTime(alert.timestamp)}</td>
                  <td style={tdStyle}>
                    <span
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize:   '11px',
                        fontWeight: 600,
                        color:      SEVERITY_COLORS[alert.severity] ?? 'var(--text)',
                      }}
                    >
                      {alert.severity}
                    </span>
                  </td>
                  <td style={tdStyle}>{alert.fall_type}</td>
                  <td style={tdStyle}>{alert.pre_activity}</td>
                  <td style={tdStyle}>{alert.post_state}</td>
                  <td style={tdStyle}>{Math.round(alert.confidence * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}

const tdStyle = {
  fontFamily:  'var(--font-mono)',
  fontSize:    '11px',
  color:       'var(--text)',
  padding:     '7px 10px',
  borderBottom:'1px solid var(--border-light)',
};
