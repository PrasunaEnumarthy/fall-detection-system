// Root layout component — CSS grid dashboard with header, live feed, context card, and history log.

import React, { useState, useEffect } from 'react';
import socket from './socket.js';
import AlertFeed   from './components/AlertFeed.jsx';
import ContextCard from './components/ContextCard.jsx';
import HistoryLog  from './components/HistoryLog.jsx';

function formatDate(date) {
  return date.toLocaleDateString('en-US', { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' });
}

const panelStyle = {
  background:  'var(--surface)',
  border:      '1px solid var(--border)',
  borderRadius:'8px',
  padding:     '20px',
  overflow:    'hidden',
};

export default function App() {
  const [isConnected,     setIsConnected]     = useState(socket.connected);
  const [initialHistory,  setInitialHistory]  = useState([]);

  // Socket connection state
  useEffect(() => {
    function onConnect()    { setIsConnected(true);  }
    function onDisconnect() { setIsConnected(false); }

    socket.on('connect',    onConnect);
    socket.on('disconnect', onDisconnect);

    return () => {
      socket.off('connect',    onConnect);
      socket.off('disconnect', onDisconnect);
    };
  }, []);

  // Load persisted history on mount
  useEffect(() => {
    fetch('/api/alerts')
      .then((r) => r.json())
      .then((data) => {
        if (data.success && Array.isArray(data.alerts)) {
          setInitialHistory(data.alerts);
        }
      })
      .catch((err) => console.warn('Could not load history:', err));
  }, []);

  return (
    <div
      style={{
        display:             'grid',
        gridTemplateRows:    'auto 1fr 1fr',
        gridTemplateColumns: '1fr 1fr',
        gap:                 '12px',
        padding:             '16px',
        height:              '100vh',
        boxSizing:           'border-box',
      }}
    >
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header
        style={{
          gridColumn:     '1 / -1',
          ...panelStyle,
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'space-between',
          padding:        '12px 20px',
        }}
      >
        {/* Left: wordmark */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div
            style={{
              width:        '3px',
              height:       '28px',
              background:   'var(--critical)',
              borderRadius: '2px',
              flexShrink:   0,
            }}
          />
          <div>
            <div
              style={{
                fontFamily:    'var(--font-display)',
                fontSize:      '18px',
                color:         'var(--text)',
                lineHeight:    1,
              }}
            >
              FALL DETECTION SYSTEM
            </div>
            <div
              style={{
                fontFamily: 'var(--font-mono)',
                fontSize:   '10px',
                color:      'var(--text-muted)',
                marginTop:  '3px',
              }}
            >
              IMU-Based Context-Aware Alert Pipeline
            </div>
          </div>
        </div>

        {/* Right: connection status + date */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
            <div
              style={{
                width:        '8px',
                height:       '8px',
                borderRadius: '50%',
                background:   isConnected ? '#22c55e' : '#ef4444',
                boxShadow:    isConnected ? '0 0 0 3px rgba(34,197,94,0.2)' : 'none',
              }}
            />
            <span
              style={{
                fontFamily:    'var(--font-mono)',
                fontSize:      '10px',
                fontWeight:    600,
                color:         isConnected ? '#16a34a' : '#dc2626',
                textTransform: 'uppercase',
                letterSpacing: '0.06em',
              }}
            >
              {isConnected ? 'LIVE' : 'DISCONNECTED'}
            </span>
          </div>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize:   '10px',
              color:      'var(--text-muted)',
            }}
          >
            {formatDate(new Date())}
          </span>
        </div>
      </header>

      {/* ── Live Alert Feed ─────────────────────────────────────────────── */}
      <div style={{ ...panelStyle, gridColumn: '1', overflow: 'hidden' }}>
        <AlertFeed isConnected={isConnected} />
      </div>

      {/* ── Context Card ───────────────────────────────────────────────── */}
      <div style={{ ...panelStyle, gridColumn: '2', overflow: 'hidden' }}>
        <ContextCard />
      </div>

      {/* ── History Log ────────────────────────────────────────────────── */}
      <div style={{ ...panelStyle, gridColumn: '1 / -1', overflow: 'hidden' }}>
        <HistoryLog initialHistory={initialHistory} />
      </div>
    </div>
  );
}
