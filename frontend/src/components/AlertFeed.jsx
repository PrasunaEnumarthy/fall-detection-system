// Live alert feed — subscribes to socket 'fall_alert' events and shows the 20 most recent alerts with slide-in animation.

import React, { useState, useEffect } from 'react';
import socket from '../socket.js';
import AlertCard from './AlertCard.jsx';

export default function AlertFeed({ isConnected }) {
  const [alerts, setAlerts]   = useState([]);
  const [newId, setNewId]     = useState(null);

  useEffect(() => {
    function handleAlert(alert) {
      const id = `${alert.timestamp}-${Math.random()}`;
      const alertWithId = { ...alert, _id: id };

      setAlerts((prev) => [alertWithId, ...prev].slice(0, 20));
      setNewId(id);
      setTimeout(() => setNewId(null), 400);
    }

    socket.on('fall_alert', handleAlert);
    return () => socket.off('fall_alert', handleAlert);
  }, []);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden' }}>
      {/* Panel header */}
      <div
        style={{
          display:        'flex',
          alignItems:     'center',
          justifyContent: 'space-between',
          marginBottom:   '14px',
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
          Live Alert Feed
        </span>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontFamily: 'var(--font-mono)',
              fontSize:   '11px',
              color:      'var(--text-dim)',
            }}
          >
            {alerts.length} / 20
          </span>
          <div
            style={{
              width:        '7px',
              height:       '7px',
              borderRadius: '50%',
              background:   isConnected ? '#22c55e' : '#ef4444',
              boxShadow:    isConnected ? '0 0 0 2px rgba(34,197,94,0.25)' : 'none',
            }}
          />
        </div>
      </div>

      {/* Alert list */}
      <div style={{ overflowY: 'auto', flexGrow: 1 }}>
        {alerts.length === 0 ? (
          <div
            style={{
              height:     '100%',
              display:    'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontFamily: 'var(--font-mono)',
              fontSize:   '12px',
              color:      'var(--text-dim)',
            }}
          >
            Monitoring active — no events yet
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {alerts.map((alert) => (
              <AlertCard
                key={alert._id}
                alert={alert}
                isNew={alert._id === newId}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
