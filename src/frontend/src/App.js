import React from "react";
import axios from 'axios';
import { useEffect, useState } from 'react';
import Chatbot from "react-chatbot-kit";
import "react-chatbot-kit/build/main.css";
import './App.css'

import config from "./config";
import MessageParser from "./MessageParser";
import ActionProvider from "./ActionProvider";

function App() {
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const checkStatus = async () => {
      try {
        const res = await axios.get('http://localhost:8000/status');
        if (res.data.ready) {
          setReady(true);
        } else {
          setTimeout(checkStatus, 1000);
        }
      } catch (err) {
        console.error(err);
        setTimeout(checkStatus, 1000);
      }
    };

    checkStatus();
  }, []);

  if (!ready) {
    return <div>모델을 로딩 중입니다...</div>;
  }

  return (
    <div className="App">
      <Chatbot
        config={config}
        messageParser={MessageParser}
        actionProvider={ActionProvider}
      />
    </div>
  );
}

export default App;
