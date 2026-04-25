import React from "react";
import "./HomePage.css";

const factors = [
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <path d="M9 3H7a2 2 0 00-2 2v14a2 2 0 002 2h10a2 2 0 002-2V5a2 2 0 00-2-2h-2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        <rect x="9" y="2" width="6" height="3" rx="1" stroke="currentColor" strokeWidth="1.8"/>
        <path d="M12 11v6M9.5 13.5L12 11l2.5 2.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
    ),
    label: "Soil pH",
    sub: "Acidity levels",
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <path d="M12 2C6 8 4 12 4 15a8 8 0 0016 0c0-3-2-7-8-13z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round"/>
      </svg>
    ),
    label: "Rainfall",
    sub: "Precipitation data",
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <path d="M12 2v10M12 22v-4M12 12a4 4 0 100 8 4 4 0 000-8z" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
        <line x1="12" y1="2" x2="12" y2="6" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"/>
      </svg>
    ),
    label: "Temperature",
    sub: "Climate conditions",
  },
  {
    icon: (
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
        <rect x="5" y="3" width="14" height="18" rx="2" stroke="currentColor" strokeWidth="1.8"/>
        <path d="M9 8h6M9 12h6M9 16h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
      </svg>
    ),
    label: "Nutrients",
    sub: "NPK values",
  },
];

export default function HomePage({ onNavigate }) {
  return (
    <div className="home">
      {/* Hero */}
      <section className="hero">
        <div className="hero__bg-leaf" aria-hidden="true">
          <svg viewBox="0 0 500 500" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M250 50C150 150 80 250 100 350C120 450 250 480 350 420C450 360 490 220 420 130C370 60 300 20 250 50Z"
              fill="rgba(255,255,255,0.06)" />
            <path d="M200 80C120 160 80 250 100 330C120 410 200 440 300 400C380 360 420 260 380 180C340 100 270 40 200 80Z"
              fill="rgba(255,255,255,0.04)" />
          </svg>
        </div>
        <div className="hero__content animate-fadeInUp">
          <h1 className="hero__title">
            Green Predict AI –<br />Tree Survival &amp;<br />Recommendation System
          </h1>
          <p className="hero__subtitle animate-fadeInUp delay-2">
            Get AI-powered survival predictions and recommendations for the best
            trees for your soil and climate in Maharashtra. Perfect for NGOs,
            government plantation programs, environmental volunteers, and
            community initiatives.
          </p>
          <button
            className="btn btn--primary btn--lg animate-fadeInUp delay-3"
            onClick={() => onNavigate("predict")}
          >
            Start Prediction
          </button>
        </div>
        <div className="hero__image-side" aria-hidden="true">
          <div className="hero__leaf-anim">🌿</div>
        </div>
      </section>

      {/* Factors */}
      <section className="factors">
        <div className="factors__card animate-scaleIn delay-2">
          <h2 className="factors__heading">Environmental Factors We Analyze</h2>
          <div className="factors__grid">
            {factors.map((f, i) => (
              <div className={`factor-item animate-fadeInUp delay-${i + 2}`} key={f.label}>
                <div className="factor-item__icon">{f.icon}</div>
                <span className="factor-item__label">{f.label}</span>
                <span className="factor-item__sub">{f.sub}</span>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
}
