// Singleton socket.io-client instance — all components share this one connection to the backend.

import { io } from 'socket.io-client';

const socket = io('http://localhost:3001', {
  reconnectionAttempts: 5,
  reconnectionDelay: 1000,
});

export default socket;
