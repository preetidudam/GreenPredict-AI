import React, { useState } from "react";
import { plantData, plantNames } from "../data/plants";
import TreeDetailModal from "../components/TreeDetailModal";
import PlantImage from "../components/PlantImage";
import "./EncyclopediaPage.css";

const SearchIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <circle cx="11" cy="11" r="7" stroke="currentColor" strokeWidth="1.8"/>
    <path d="M21 21l-4.35-4.35" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
  </svg>
);

const LeafIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
    <path d="M17 8C8 10 5.9 16.17 3.82 19.57C3.82 19.57 8.33 20.5 12 18C14.78 16.3 17.5 13 17 8Z" fill="currentColor" opacity="0.8"/>
    <path d="M21 3C19 8 14.5 10 11 12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
  </svg>
);

export default function EncyclopediaPage() {
  const [search, setSearch] = useState("");
  const [detailPlant, setDetailPlant] = useState(null);

  const filtered = plantNames.filter((p) =>
    p.toLowerCase().includes(search.toLowerCase()) ||
    plantData[p].scientificName.toLowerCase().includes(search.toLowerCase()) ||
    plantData[p].benefits.some((b) => b.toLowerCase().includes(search.toLowerCase()))
  );

  return (
    <div className="encyclopedia">
      <div className="encyclopedia__container">
        {/* Header */}
        <div className="encyclopedia__header animate-fadeInUp">
          <h1 className="encyclopedia__title">
            <span className="encyclopedia__title-icon"><LeafIcon /></span>
            Tree Encyclopedia
          </h1>
          <p className="encyclopedia__subtitle">
            Comprehensive guide for the {plantNames.length} recommended trees suitable for Maharashtra region
          </p>
          <div className="search-wrap">
            <span className="search-icon"><SearchIcon /></span>
            <input
              className="search-input"
              type="text"
              placeholder="Search for trees..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
        </div>

        {/* Grid */}
        {filtered.length > 0 ? (
          <div className="enc-grid">
            {filtered.map((plant, i) => {
              const info = plantData[plant];
              return (
                <div
                  key={plant}
                  className={`enc-card animate-fadeInUp delay-${Math.min(i + 1, 6)}`}
                  onClick={() => setDetailPlant(plant)}
                >
                  <div className="enc-card__img-wrap">
                    <PlantImage plant={plant} className="enc-card__img" alt={plant} />
                  </div>
                  <div className="enc-card__body">
                    <h3 className="enc-card__name">{plant}</h3>
                    <div className="enc-card__row">
                      <span className="enc-card__row-icon">🌱</span>
                      <div>
                        <span className="enc-card__row-label">Soil</span>
                        <span className="enc-card__row-val">{info.soil}</span>
                      </div>
                    </div>
                    <div className="enc-card__row">
                      <span className="enc-card__row-icon">☀️</span>
                      <div>
                        <span className="enc-card__row-label">Climate</span>
                        <span className="enc-card__row-val">{info.climate}</span>
                      </div>
                    </div>
                    <div className="enc-card__row">
                      <span className="enc-card__row-icon">✨</span>
                      <div>
                        <span className="enc-card__row-label">Benefits</span>
                        <span className="enc-card__row-val enc-card__row-val--green">
                          {info.benefits.join(", ")}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="enc-empty animate-fadeIn">
            <span>🌿</span>
            <p>No trees found for "<strong>{search}</strong>"</p>
          </div>
        )}

        {/* About */}
        <div className="about-card animate-fadeInUp delay-4">
          <h3 className="about-card__heading">🌳 About These Trees</h3>
          <p className="about-card__body">
            These {plantNames.length} trees are specifically selected for Maharashtra's climate and soil
            conditions. They are suitable for large-scale plantation in gardens, parks, roadsides, and
            environmental programs. They provide significant environmental benefits including carbon
            sequestration, soil conservation, shade, medicinal properties, and economic value. Ideal for
            NGOs, government plantation programs, environmental volunteers, and community initiatives.
          </p>
        </div>
      </div>

      {detailPlant && (
        <TreeDetailModal plant={detailPlant} onClose={() => setDetailPlant(null)} />
      )}
    </div>
  );
}
