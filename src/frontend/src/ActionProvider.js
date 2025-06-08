class ActionProvider {
    constructor(createChatBotMessage, setStateFunc) {
        this.createChatBotMessage = createChatBotMessage;
        this.setState = setStateFunc;
    }

    async handleUserMessage(userMessage) {
        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            });

            const data = await response.json();

            const botMessage = this.createChatBotMessage(data.len);
            this.setState((prev) => ({
                ...prev,
                messages: [...prev.messages, botMessage],
            }));
        } catch (error) {
            const errorMsg = this.createChatBotMessage("서버 오류가 발생했습니다.");
            this.setState((prev) => ({
                ...prev,
                messages: [...prev.messages, errorMsg],
            }));
        }
    }
}

export default ActionProvider;