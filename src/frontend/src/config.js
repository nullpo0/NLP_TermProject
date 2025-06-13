import { createChatBotMessage } from "react-chatbot-kit";

const config = {
    botName: "차차 chatbot",
    initialMessages: [
        createChatBotMessage("안녕하세요! 차차 챗봇입니다. 무엇을 도와드릴까요?"),
    ],
    customComponents: {
        header: () => null,
        botAvatar: () => <div className="custom-bot-avatar">
      <img src="/images/chacha.png" alt="bot" style={{ width: 65, height: 65, borderRadius: '50%' }} />
    </div>
    },
    customStyles: {
        botMessageBox: {
            backgroundColor: "#00aff0;",
        }
    },
};

export default config;
