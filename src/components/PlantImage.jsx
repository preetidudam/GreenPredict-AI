import React, { useState } from "react";
import { plantData } from "../data/plants";
import plantImages from "../assets/images/index.js";

/**
 * PlantImage — shows the locally uploaded image first.
 * Falls back to remote URLs in plantData[plant].images if local fails.
 * Final fallback: styled emoji placeholder.
 */
export default function PlantImage({ plant, className = "", alt, style = {} }) {
  const info = plantData[plant];
  const localSrc = plantImages[plant];
  const remoteSources = info?.images || [];

  // local image first, then remote fallbacks
  const allSources = localSrc
    ? [localSrc, ...remoteSources]
    : remoteSources;

  const [idx, setIdx] = useState(0);
  const [failed, setFailed] = useState(false);

  const handleError = () => {
    if (idx < allSources.length - 1) {
      setIdx((i) => i + 1);
    } else {
      setFailed(true);
    }
  };

  if (failed || allSources.length === 0) {
    return (
      <div
        className={className}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: `linear-gradient(135deg, ${info?.color || "#2d6a35"}22, ${info?.color || "#2d6a35"}44)`,
          fontSize: "3.5rem",
          ...style,
        }}
        aria-label={alt || plant}
      >
        {info?.emoji || "🌳"}
      </div>
    );
  }

  return (
    <img
      className={className}
      src={allSources[idx]}
      alt={alt || plant}
      onError={handleError}
      style={style}
      loading="lazy"
    />
  );
}
