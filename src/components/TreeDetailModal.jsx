import React from "react";
import PlantImage from "./PlantImage";
import { plantData } from "../data/plants";
import "./TreeDetailModal.css";

export default function TreeDetailModal({ plant, onClose }) {
  if (!plant) return null;
  const info = plantData[plant];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <button className="modal__close" onClick={onClose} aria-label="Close">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6l12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
          </svg>
        </button>

        <div className="modal__img-wrap">
          <PlantImage plant={plant} className="modal__img" alt={plant} />
          <div className="modal__img-overlay">
            <h2 className="modal__img-name">{plant}</h2>
            <p className="modal__img-sci">{info.scientificName}</p>
          </div>
        </div>

        <div className="modal__body">
          <p className="modal__desc">{info.description}</p>

          <div className="modal__rows">
            <div className="modal__row">
              <span className="modal__row-icon">🌱</span>
              <div>
                <span className="modal__row-label">Soil</span>
                <span className="modal__row-val">{info.soil}</span>
              </div>
            </div>
            <div className="modal__row">
              <span className="modal__row-icon">☀️</span>
              <div>
                <span className="modal__row-label">Climate</span>
                <span className="modal__row-val">{info.climate}</span>
              </div>
            </div>
            <div className="modal__row">
              <span className="modal__row-icon">✨</span>
              <div>
                <span className="modal__row-label">Benefits</span>
                <div className="modal__tags">
                  {info.benefits.map((b) => (
                    <span key={b} className="modal__tag">{b}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
