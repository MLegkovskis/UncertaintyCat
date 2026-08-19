import { Bot, Send, User } from "lucide-react";
import { useEffect, useState } from "react";

import { api, readTextStream } from "../api";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export function ChatPanel({ reportId }: { reportId: string }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let active = true;
    void api
      .getChatMessages(reportId)
      .then(({ messages: stored }) => {
        if (active)
          setMessages(stored.map(({ role, content }) => ({ role, content })));
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [reportId]);

  async function send() {
    const message = input.trim();
    if (!message || sending) return;
    setInput("");
    setError(undefined);
    setSending(true);
    setMessages((current) => [
      ...current,
      { role: "user", content: message },
      { role: "assistant", content: "" },
    ]);
    try {
      const response = await fetch(`/api/v1/reports/${reportId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ message }),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => ({}))) as {
          error?: { message?: string };
        };
        throw new Error(body.error?.message ?? "Chat is unavailable.");
      }
      await readTextStream(response, (chunk) =>
        setMessages((current) =>
          current.map((item, index) =>
            index === current.length - 1
              ? { ...item, content: item.content + chunk }
              : item,
          ),
        ),
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Chat failed.");
      setMessages((current) => current.slice(0, -1));
    } finally {
      setSending(false);
    }
  }

  return (
    <aside className="chat-panel">
      <div className="chat-heading">
        <span>
          <Bot size={18} />
        </span>
        <div>
          <strong>Ask this report</strong>
          <small>
            Answers use stored facts and cite their analysis source.
          </small>
        </div>
      </div>
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-suggestions">
            <p>Try asking:</p>
            {[
              "Which input has the greatest influence?",
              "Compare the sensitivity findings.",
              "What assumptions should I review?",
            ].map((suggestion) => (
              <button key={suggestion} onClick={() => setInput(suggestion)}>
                {suggestion}
              </button>
            ))}
          </div>
        )}
        {messages.map((message, index) => (
          <div className={`chat-message ${message.role}`} key={index}>
            {message.role === "user" ? <User /> : <Bot />}
            <p>{message.content || "Thinking…"}</p>
          </div>
        ))}
      </div>
      {error && <p className="inline-error">{error}</p>}
      <div className="chat-input">
        <textarea
          aria-label="Question about report"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void send();
            }
          }}
          placeholder="Ask about a result…"
          rows={2}
        />
        <button
          onClick={() => void send()}
          disabled={sending || !input.trim()}
          aria-label="Send question"
        >
          <Send />
        </button>
      </div>
      <small className="ai-label">
        AI-generated explanation. Verify decisions against the numerical tables.
      </small>
    </aside>
  );
}
