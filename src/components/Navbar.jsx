import React, { useState, useEffect } from "react";
import "./Navbar.css";

const LeafIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M21 3C19 8 14.5 10 11 12C7.5 14 5 17 3.82 19.57" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
    <path d="M17 8C8 10 5.9 16.17 3.82 19.57C3.82 19.57 8.33 20.5 12 18C14.78 16.3 17.5 13 17 8Z"
      fill="currentColor" opacity="0.85"/>
    <path d="M3.82 19.57L3 21" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/>
  </svg>
);

export default function Navbar({ activePage, onNavigate }) {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav className={`navbar ${scrolled ? "navbar--scrolled" : ""}`}>
      <div className="navbar__inner">
        <button className="navbar__brand" onClick={() => onNavigate("home")}>
          <span className="navbar__logo-wrap">
            <LeafIcon />
          </span>
          <span className="navbar__brand-name">GreenPredict-AI</span>
        </button>

        <div className="navbar__links">
          <button
            className={`navbar__link ${activePage === "home" ? "navbar__link--active" : ""}`}
            onClick={() => onNavigate("home")}
          >
            Home
          </button>
          <button
            className={`navbar__link ${activePage === "encyclopedia" ? "navbar__link--active" : ""}`}
            onClick={() => onNavigate("encyclopedia")}
          >
            Tree Encyclopedia
          </button>
        </div>
      </div>
    </nav>
  );
}
