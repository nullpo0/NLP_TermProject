import { createChatBotMessage } from "react-chatbot-kit";

const config = {
    botName: "CNU chatbot",
    initialMessages: [
        createChatBotMessage("안녕하세요! CNU 챗봇입니다. 무엇을 도와드릴까요?"),
    ],
    customStyles: {
        botMessageBox: {
            backgroundColor: "#376B7E",
        },
        chatButton: {
            backgroundColor: "#5ccc9d",
        },
    },
};

export default config;
