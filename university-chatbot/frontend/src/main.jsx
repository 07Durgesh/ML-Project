import React from 'react';
import ReactDOM from 'react-dom/client';
import UniversityChatbot from './UniversityChatbot.jsx'; // The main app component
import './index.css';                                 // Your main stylesheet

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <UniversityChatbot />
  </React.StrictMode>
);