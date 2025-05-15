import React from 'react';
import { Link } from 'react-router-dom';
import './Home.css';

export default function Home() {
  return (
    <div className="home-container">
      <div className="home-header">
        <h1 className="home-heading">Yoga Pose Detection</h1>
      </div>
      <div className="home-main">
        <h1 className="description">Transform Your Yoga Practice with AI</h1>
        <p className="sub-description">
          Achieve perfect poses with real-time feedback and guidance.
        </p>
        <div className="btn-section">
          <Link to="/start">
            <button className="btn start-btn">Let's Begin</button>
          </Link>
        </div>
      </div>
    </div>
  );
}