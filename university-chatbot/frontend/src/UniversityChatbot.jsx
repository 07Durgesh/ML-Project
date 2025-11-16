import React, { useState, useEffect, useRef } from 'react';
import { Send, Bot, User, Loader2, MessageSquare, Download } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:8000'; // Port updated to 8000

// A simple component to render markdown-style bold text (**text**)
const MarkdownRenderer = ({ text }) => {
    if (typeof text !== 'string') {
        return null;
    }
    const parts = text.split(/(\*\*.*?\*\*)/g);
    return (
        <p className="whitespace-pre-wrap text-sm leading-relaxed">
            {parts.map((part, index) => {
                if (part.startsWith('**') && part.endsWith('**')) {
                    return <strong key={index}>{part.slice(2, -2)}</strong>;
                }
                return part;
            })}
        </p>
    );
};


const UniversityChatbot = () => {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      text: "Hello! I'm your AI assistant. Ask me anything about university courses, professors, deadlines, and more!",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId] = useState(`session_${Date.now()}`);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  const handleSendMessage = async (messageText = input) => {
    if (!messageText.trim() || isLoading) return;

    const userMessage = { type: 'user', text: messageText, timestamp: new Date().toISOString() };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageText, session_id: sessionId }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      const botMessage = {
        type: 'bot',
        text: data.response,
        timestamp: data.timestamp,
        data: data.data,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Fetch error:", error);
      const errorMessage = {
        type: 'bot',
        text: '❌ Sorry, an error occurred. Please check the backend server and try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTimestamp = (isoString) => {
    return new Date(isoString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const suggestivePretexts = [
    "What is the syllabus for CSET301 Machine Learning?",
    "Show me past year questions for Data Structures",
    "What are the upcoming deadlines?",
    "Who is the professor for Databases?",
    "Where is the Library?",
    "Contact details for Dr. Vikram Singh"
  ];

  return (
    <div className="relative flex h-screen w-full items-center justify-center bg-primary-dark font-sans text-text-light">
      <div className="absolute inset-0 bg-[radial-gradient(#333a4d_1px,transparent_1px)] [background-size:16px_16px] [mask-image:radial-gradient(ellipse_50%_50%_at_50%_50%,#000_70%,transparent_100%)] opacity-30"></div>
      <div className="relative z-10 flex h-[90vh] w-full max-w-2xl flex-col rounded-xl bg-secondary-dark shadow-2xl backdrop-blur-sm md:h-[80vh]">
        <div className="flex items-center justify-center border-b border-gray-700 p-6">
          <MessageSquare className="h-8 w-8 text-accent-green mr-3" />
          <div>
            <h1 className="text-3xl font-extrabold tracking-tight">BU Buddy</h1>
            <p className="text-sm text-text-muted">Your intelligent campus companion</p>
          </div>
        </div>
        <div className="flex-1 space-y-4 overflow-y-auto p-6 scrollbar-dark">
          {messages.map((message, index) => (
            <div key={index} className="flex flex-col animate-fade-in-up">
              <div className={`max-w-[85%] rounded-lg p-3 shadow-md ${message.type === 'user' ? 'self-end rounded-br-none bg-blue-600' : 'self-start rounded-bl-none bg-gray-700'}`}>
                <MarkdownRenderer text={message.text} />
                {message.type === 'bot' && message.data && message.data.pdf_url && (
                  <a
                    href={`${API_URL}/static/${message.data.pdf_url}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="mt-3 inline-flex items-center gap-2 rounded-lg bg-accent-green/20 px-3 py-2 text-xs font-semibold text-accent-green transition-all hover:bg-accent-green/40"
                  >
                    <Download className="h-4 w-4" />
                    Download Syllabus PDF
                  </a>
                )}
              </div>
              <p className={`mt-1 text-xs text-gray-500 ${message.type === 'user' ? 'self-end' : 'self-start'}`}>
                {formatTimestamp(message.timestamp)}
              </p>
            </div>
          ))}
          {isLoading && (
            <div className="self-start rounded-lg bg-gray-700 p-3 shadow-md">
              <Loader2 className="h-5 w-5 animate-spin text-accent-green" />
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        {messages.length === 1 && (
          <div className="flex flex-wrap justify-center gap-2 p-4 border-t border-gray-700 bg-secondary-dark">
            {suggestivePretexts.map((text, index) => (
              <button
                key={index}
                onClick={() => handleSendMessage(text)}
                className="flex items-center space-x-2 rounded-full bg-gray-700 px-4 py-2 text-sm text-text-light transition-all hover:bg-gray-600 hover:text-accent-green focus:outline-none focus:ring-2 focus:ring-accent-green focus:ring-offset-2 focus:ring-offset-secondary-dark"
              >
                <MessageSquare className="h-4 w-4" />
                <span>{text}</span>
              </button>
            ))}
          </div>
        )}
        <div className="border-t border-gray-700 p-4">
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message..."
              className="flex-1 rounded-lg border-none bg-gray-700 px-4 py-3 text-text-light placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-accent-green"
              disabled={isLoading}
            />
            <button
              onClick={() => handleSendMessage()}
              disabled={!input.trim() || isLoading}
              className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-accent-green text-primary-dark transition-colors hover:bg-green-600 disabled:cursor-not-allowed disabled:bg-gray-600"
            >
              <Send className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UniversityChatbot;