import React, { useState } from "react";
import "./styles/globals.css";
import Navbar from "./components/Navbar";
import HomePage from "./pages/HomePage";
import PredictPage from "./pages/PredictPage";
import ResultsPage from "./pages/ResultsPage";
import EncyclopediaPage from "./pages/EncyclopediaPage";
import ChatPage from "./pages/ChatPage";

export default function App() {
  const [page, setPage] = useState("home");
  const [results, setResults] = useState(null);

  const navigate = (target) => {
    setPage(target);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="app">
      <Navbar activePage={page} onNavigate={navigate} />
      {page === "home"         && <HomePage onNavigate={navigate} />}
      {page === "predict"      && <PredictPage onNavigate={navigate} onResults={setResults} />}
      {page === "results"      && <ResultsPage results={results} onNavigate={navigate} />}
      {page === "encyclopedia" && <EncyclopediaPage />}
      {page === "chat"         && <ChatPage />}
    </div>
  );
}
