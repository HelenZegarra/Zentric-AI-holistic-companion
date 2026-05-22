import { useState, useEffect, useRef } from 'react';
import './App.css';
import Particles from 'react-tsparticles';
import type { Container, Engine } from 'tsparticles-engine';
import { loadSlim } from 'tsparticles-slim';
import ReactMarkdown from 'react-markdown';

interface Message {
  content: string;
  sender: 'user' | 'zentric';
}

interface AgentState {
  user_id: string;
  chat_history: Message[];
  user_profile: {
    conditions: string[];
    dietary_preferences: string[];
  };
  current_plan: {};
  rewards_points: number;
  plan_generated_today: boolean;
  comparison_query: string;
  glucose_trend_data: {};
  wellness_sparks: any[];
  zentric_whisper: string;
  input: string;
  loop_count: number;
  current_stage: string;
  glucose_level: number | null;
  heart_rate: number | null;
  steps: number | null;
  sleep_hours: number | null;
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [agentState, setAgentState] = useState<AgentState | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  const particlesInit = async (engine: Engine) => {
    await loadSlim(engine);
  };

  const particlesLoaded = async (container: Container | undefined) => {
    console.log(container);
  };

  useEffect(() => {
    setMessages([
      {
        content: "Hello there! 👋 I'm Zentric, your personal wellness agent. Let's start with your health profile. Can you tell me about any conditions (like diabetes) or dietary preferences you have?",
        sender: 'zentric',
      },
    ]);
  }, []);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (inputValue.trim() === '') return;

    const newUserMessage: Message = { content: inputValue, sender: 'user' };
    setMessages(prevMessages => [...prevMessages, newUserMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ input: inputValue, agentState: agentState }),
      });

      const data = await response.json();
      if (response.ok) {
        const zentricResponse: Message = { content: data.response, sender: 'zentric' };
        setMessages(prevMessages => [...prevMessages, zentricResponse]);
        setAgentState(data.agentState);
      } else {
        setMessages(prevMessages => [
          ...prevMessages,
          { content: `Error: ${data.error}`, sender: 'zentric' },
        ]);
        console.error('API Error:', data.error);
      }
    } catch (error) {
      setMessages(prevMessages => [
        ...prevMessages,
        { content: `An unexpected error occurred: ${error}`, sender: 'zentric' },
      ]);
      console.error('Fetch Error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSendMessage();
    }
  };

  return (
    <>
      <Particles
        id="tsparticles"
        init={particlesInit}
        loaded={particlesLoaded}
        options={{
          background: {
            color: {
              value: "transparent",
            },
          },
          fpsLimit: 120,
          interactivity: {
            events: {
              onClick: {
                enable: true,
                mode: "push",
              },
              onHover: {
                enable: true,
                mode: "repulse",
              },
              resize: true,
            },
            modes: {
              push: {
                quantity: 4,
              },
              repulse: {
                distance: 200,
                duration: 0.4,
              },
            },
          },
          particles: {
            color: {
              value: "#ffffff",
            },
            links: {
              color: "#ffffff",
              distance: 150,
              enable: true,
              opacity: 0.5,
              width: 1,
            },
            move: {
              direction: "none",
              enable: true,
              outModes: {
                default: "bounce",
              },
              random: false,
              speed: 6,
              straight: false,
            },
            number: {
              density: {
                enable: true,
                area: 800,
              },
              value: 80,
            },
            opacity: {
              value: 0.5,
            },
            shape: {
              type: "circle",
            },
            size: {
              value: { min: 1, max: 5 },
            },
          },
          detectRetina: true,
        }}
      />
      <div className="app-container">
        <div className="app-header">
          <h1 className="app-title">Zentric</h1>
        </div>
        <div className="chat-messages">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${message.sender}-message`}
            >
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          ))}
          {isLoading && (
            <div className="typing-indicator zentric-message">
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
              <div className="typing-dot"></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        <div className="chat-input-area">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
          />
          <button onClick={handleSendMessage} disabled={isLoading}>
            Send
          </button>
        </div>
      </div>
    </>
  );
}

export default App;