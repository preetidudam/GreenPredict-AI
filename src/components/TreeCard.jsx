import React from "react";
import PlantImage from "./PlantImage";
import "./TreeCard.css";

const EyeIcon = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" stroke="currentColor" strokeWidth="1.8"/>
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.8"/>
  </svg>
);

export default function TreeCard({ plant, probability, rank, isSelected, onViewDetails }) {
  return (
    <div className={`tree-card ${isSelected ? "tree-card--selected" : ""} animate-scaleIn`}>
      {/* Image */}
      <div className="tree-card__img-wrap">
        <PlantImage plant={plant} className="tree-card__img" alt={plant} />
        <span className="tree-card__rank">#{rank}</span>
        {isSelected && <span className="tree-card__badge">Your Choice</span>}
        {rank === 1 && !isSelected && <span className="tree-card__top">Top Pick</span>}
      </div>

      {/* Body */}
      <div className="tree-card__body">
        <div className="tree-card__header">
          <div>
            <h3 className="tree-card__name">{plant}</h3>
            <p className="tree-card__sci">
              {require("../data/plants").plantData[plant].scientificName}
            </p>
          </div>
          <div className="tree-card__pct-wrap">
            <span className="tree-card__pct-arrow">↗</span>
            <span className="tree-card__pct">{probability.toFixed(0)}%</span>
            <span className="tree-card__pct-label">Survival</span>
          </div>
        </div>

        <p className="tree-card__desc">
          {require("../data/plants").plantData[plant].description}
        </p>

        <div className="tree-card__benefits">
          <span className="tree-card__benefits-label">Benefits:</span>
          <span className="tree-card__benefits-list">
            {require("../data/plants").plantData[plant].benefits.join(", ")}
          </span>
        </div>

        <button className="tree-card__btn" onClick={() => onViewDetails(plant)}>
          <EyeIcon /> View Details
        </button>
      </div>
    </div>
  );
}
