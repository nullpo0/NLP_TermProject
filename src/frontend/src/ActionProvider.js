class ActionProvider {
    constructor(createChatBotMessage, setStateFunc) {
        this.createChatBotMessage = createChatBotMessage;
        this.setState = setStateFunc;
    }

    async handleUserMessage(userMessage) {
        const loadingMessage = this.createChatBotMessage("답변을 생성 중입니다...");

        let loadingMessageId;
        this.setState((prev) => {
            loadingMessageId = prev.messages.length;
            return {
                ...prev,
                messages: [...prev.messages, loadingMessage],
            };
        });

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: userMessage }),
            });

            const data = await response.json();

            const botMessage = this.createChatBotMessage(data.answer);
            this.setState((prev) => {
                const newMessages = [...prev.messages];
                newMessages[loadingMessageId] = botMessage;
                return {
                    ...prev,
                    messages: newMessages,
                };
            });
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