import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [message, setMessage] = useState('');

  const handleClick = async () => {
    try {
      const response = await axios.get('http://localhost:8000/response');
      setMessage(response.data.response);
    } catch (error) {
      console.error('백엔드 요청 실패:', error);
      setMessage('요청 실패');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <button onClick={handleClick}>서버에 요청 보내기</button>
      <p>{message}</p>
    </div>
  );
}

export default App;
