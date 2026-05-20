import React, { useState, useRef, useEffect, useCallback } from "react";
import "./ChatPage.css";

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

const SUGGESTIONS = [
  "What is GreenPredict-AI?",
  "Which tree species does it support?",
  "What soil parameters are needed?",
  "How does the ML model work?",
  "Who can use this project?",
];

// ---------------------------------------------------------------------------
// API call — returns { answer } or throws with isRateLimit flag
// ---------------------------------------------------------------------------
async function fetchAnswer(question) {
  const res  = await fetch(`${BASE_URL}/chat`, {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ question }),
  });
  const json = await res.json();

  if (res.status === 429 || json.rate_limited) {
    const err      = new Error(json.error || "Rate limited.");
    err.isRateLimit = true;
    throw err;
  }
  if (!res.ok) throw new Error(json.error || "Something went wrong.");
  return json.answer;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export default function ChatPage() {
  const [messages,   setMessages]   = useState([
    { role: "bot", text: "Hello! I'm the GreenPredict-AI assistant. Ask me anything about the project — what it does, how it works, which trees are supported, and more." },
  ]);
  const [input,      setInput]      = useState("");
  const [loading,    setLoading]    = useState(false);

  // Countdown state for rate-limit retries
  const [countdown,  setCountdown]  = useState(0);   // seconds remaining
  const [pendingQ,   setPendingQ]   = useState(null); // question waiting to retry

  const bottomRef   = useRef(null);
  const inputRef    = useRef(null);
  const timerRef    = useRef(null);

  // Auto-scroll
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, countdown]);

  // Countdown tick — auto-retries when it reaches 0
  useEffect(() => {
    if (countdown <= 0) return;

    timerRef.current = setTimeout(() => {
      setCountdown((c) => {
        if (c <= 1) {
          // Time's up — auto-retry the pending question
          if (pendingQ) {
            setPendingQ(null);
            doSend(pendingQ, true); // isRetry = true
          }
          return 0;
        }
        return c - 1;
      });
    }, 1000);

    return () => clearTimeout(timerRef.current);
  }, [countdown, pendingQ]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------------------------------------------------------------------------
  // Core send function
  // ---------------------------------------------------------------------------
  const doSend = useCallback(async (question, isRetry = false) => {
    if (!question || loading) return;

    if (!isRetry) {
      setMessages((m) => [...m, { role: "user", text: question }]);
      setInput("");
    }
    setLoading(true);
    setCountdown(0);

    try {
      const answer = await fetchAnswer(question);
      setMessages((m) => [...m, { role: "bot", text: answer }]);
      setPendingQ(null);
    } catch (err) {
      if (err.isRateLimit) {
        // Show a rate-limit info bubble and start countdown for auto-retry
        setPendingQ(question);
        setCountdown(30);
        setMessages((m) => [
          ...m,
          {
            role:        "bot",
            text:        "⏳ The AI is temporarily busy (free-tier limit). Auto-retrying in 30 seconds…",
            isRateLimit: true,
          },
        ]);
      } else {
        setMessages((m) => [
          ...m,
          { role: "bot", text: err.message, isError: true },
        ]);
      }
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }, [loading]);

  const sendMessage = (text) => {
    const q = (text || input).trim();
    if (!q || loading || countdown > 0) return;
    doSend(q);
  };

  const handleKey = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const isBusy = loading || countdown > 0;

  return (
    <div className="chat-page">
      <div className="chat-page__container">

        {/* Header */}
        <div className="chat-header animate-fadeInUp">
          <div className="chat-header__icon">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"
                fill="currentColor" opacity="0.15" stroke="currentColor"
                strokeWidth="1.8" strokeLinejoin="round"/>
            </svg>
          </div>
          <div>
            <h1 className="chat-header__title">Ask AI</h1>
            <p className="chat-header__sub">Answers based on project documentation</p>
          </div>
        </div>

        {/* Chat window */}
        <div className="chat-window animate-scaleIn delay-1">

          {/* Message list */}
          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div
                key={i}
                className={`chat-msg chat-msg--${msg.role}${
                  msg.isError     ? " chat-msg--error"      : ""
                }${msg.isRateLimit ? " chat-msg--ratelimit" : ""}`}
              >
                {msg.role === "bot" && (
                  <span className="chat-msg__avatar">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                      <path d="M21 3C19 8 14.5 10 11 12C7.5 14 5 17 3.82 19.57"
                        stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                      <path d="M17 8C8 10 5.9 16.17 3.82 19.57C3.82 19.57 8.33 20.5 12 18C14.78 16.3 17.5 13 17 8Z"
                        fill="currentColor" opacity="0.85"/>
                    </svg>
                  </span>
                )}
                <div className="chat-msg__bubble">
                  <p className="chat-msg__text">{msg.text}</p>
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {loading && (
              <div className="chat-msg chat-msg--bot">
                <span className="chat-msg__avatar">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                    <path d="M21 3C19 8 14.5 10 11 12C7.5 14 5 17 3.82 19.57"
                      stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/>
                    <path d="M17 8C8 10 5.9 16.17 3.82 19.57C3.82 19.57 8.33 20.5 12 18C14.78 16.3 17.5 13 17 8Z"
                      fill="currentColor" opacity="0.85"/>
                  </svg>
                </span>
                <div className="chat-msg__bubble chat-msg__bubble--typing">
                  <span className="dot" />
                  <span className="dot" />
                  <span className="dot" />
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          {/* Countdown banner */}
          {countdown > 0 && (
            <div className="chat-countdown-bar">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2"/>
                <path d="M12 6v6l4 2" stroke="currentColor" strokeWidth="2"
                  strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
              Auto-retrying in <strong>{countdown}s</strong>…
              <button className="chat-countdown-cancel" onClick={() => {
                clearTimeout(timerRef.current);
                setCountdown(0);
                setPendingQ(null);
              }}>
                Cancel
              </button>
            </div>
          )}

          {/* Suggestions */}
          {messages.length === 1 && !isBusy && (
            <div className="chat-suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="chat-suggestion-btn"
                  onClick={() => sendMessage(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          {/* Input bar */}
          <div className="chat-input-bar">
            <input
              ref={inputRef}
              className="chat-input"
              type="text"
              placeholder={
                countdown > 0
                  ? `Auto-retrying in ${countdown}s…`
                  : "Ask about the project..."
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKey}
              disabled={isBusy}
              maxLength={500}
            />
            <button
              className="chat-send-btn"
              onClick={() => sendMessage()}
              disabled={!input.trim() || isBusy}
              aria-label="Send"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2"
                  strokeLinecap="round" strokeLinejoin="round"/>
                <path d="M22 2L15 22 11 13 2 9l20-7z" stroke="currentColor"
                  strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
