import axios from "axios";
import React, { useState } from "react";
import ReactMarkdown from 'react-markdown';

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const API_URL = import.meta.env.VITE_API_URL || "https://convocraft-backend-u1do.onrender.com";

  const sendMessage = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const res = await axios.post(`${API_URL}/chat`, { message: input });
      const botMessage = { sender: "bot", text: res.data.response };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Error communicating with the chatbot", {
        status: error.response?.status,
        data: error.response?.data,
        message: error.message,
      });
    
      const errorMessage = {
        sender: "bot",
        text: "Something went wrong. Please try again.",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div 
      id="chat-section"
      className="relative justify-center w-full bg-n-8 text-n-1 rounded-lg shadow-lg overflow-hidden h-[500px] flex flex-col"
    >
      <header className="p-4 bg-n-7 text-center text-lg font-semibold">
        Chat with GraphMind, AI Assistant powered by LangGraph and GPT!
      </header>
      <div className="flex-1 p-4 overflow-y-auto space-y-3">
        {messages.length === 0 && (
          <div className="text-center text-n-3">
            Start a conversation...
          </div>
        )}
        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${
              message.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`inline-block px-4 py-2 rounded-lg ${
                message.sender === "user"
                  ? "bg-color-1 text-n-1"
                  : "bg-n-9 text-n-1"
              }`}
            >
              {message.sender === "user" ? (
                message.text
              ) : (
                <ReactMarkdown
                  className="prose prose-invert prose-p:leading-normal prose-p:my-0 prose-ul:my-0 prose-ul:list-disc prose-ul:pl-4"
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    ul: ({ children }) => <ul className="mb-2 last:mb-0">{children}</ul>,
                    li: ({ children }) => <li className="mb-1 last:mb-0">{children}</li>,
                  }}
                >
                  {message.text}
                </ReactMarkdown>
              )}
            </div>
          </div>
        ))}
      </div>
      <footer className="p-4 bg-n-7 flex items-center space-x-2">
        <input
          type="text"
          className="flex-grow px-4 py-2 rounded-lg bg-n-6 text-n-1 focus:outline-none"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
        />
        <button
          className="px-4 py-2 bg-color-2 text-n-8 font-semibold rounded-lg hover:bg-color-3 transition"
          onClick={sendMessage}
        >
          Send
        </button>
      </footer>
    </div>
  );
};

export default Chat;
