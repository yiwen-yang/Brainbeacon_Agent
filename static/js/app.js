// Markdown 渲染配置
if (window.marked) {
    marked.setOptions({ breaks: true });
}

// 全局变量
let currentSessionId = 'default';
let isWaitingResponse = false;
let isComposing = false;

// DOM 元素
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const newChatBtn = document.getElementById('newChatBtn');

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initializeSession();
    setupEventListeners();
    adjustTextareaHeight();
});

// 初始化会话
async function initializeSession() {
    try {
        const response = await fetch('/api/new_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        currentSessionId = data.session_id;
    } catch (error) {
        console.error('初始化会话失败:', error);
    }
}

// 设置事件监听
function setupEventListeners() {
    // 发送按钮
    sendBtn.addEventListener('click', handleSend);
    
    // 输入框事件
    messageInput.addEventListener('input', () => {
        adjustTextareaHeight();
        updateSendButton();
    });
    messageInput.addEventListener('compositionstart', () => {
        isComposing = true;
    });
    messageInput.addEventListener('compositionend', () => {
        isComposing = false;
        updateSendButton();
    });
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
            e.preventDefault();
            if (!isWaitingResponse && messageInput.value.trim()) {
                handleSend();
            }
        }
    });
    
    // 新对话按钮
    newChatBtn.addEventListener('click', handleNewChat);
}

// 调整文本区域高度
function adjustTextareaHeight() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
}

// 更新发送按钮状态
function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isWaitingResponse;
}

function createAvatar(role) {
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';

    const img = document.createElement('img');
    img.src = role === 'user'
        ? '/static/img/user_avatar.png'
        : '/static/img/logo.svg';
    img.alt = role === 'user' ? '用户头像' : 'BrainBeacon 标识';
    avatar.appendChild(img);

    return avatar;
}

// 处理发送消息
async function handleSend() {
    const message = messageInput.value.trim();
    if (!message || isWaitingResponse) return;
    
    // 清空输入框
    messageInput.value = '';
    adjustTextareaHeight();
    updateSendButton();
    
    // 移除欢迎消息
    const welcomeMsg = chatContainer.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // 添加用户消息
    addMessage('user', message);
    
    // 显示加载指示器
    const loadingId = addLoadingIndicator();
    
    // 设置等待状态
    isWaitingResponse = true;
    updateSendButton();
    
    try {
        // 发送请求
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: currentSessionId
            })
        });
        
        if (!response.ok) {
            throw new Error('请求失败');
        }
        
        const data = await response.json();
        
        // 移除加载指示器
        removeLoadingIndicator(loadingId);
        
        // 添加助手回复
        addMessage('assistant', data.response);
        
    } catch (error) {
        console.error('发送消息失败:', error);
        removeLoadingIndicator(loadingId);
        addMessage('assistant', '抱歉，发生了错误。请稍后重试。');
    } finally {
        isWaitingResponse = false;
        updateSendButton();
    }
}

// 添加消息到聊天容器
function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = createAvatar(role);
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    if (role === 'assistant' && window.marked && window.DOMPurify) {
        const html = marked.parse(content);
        messageContent.innerHTML = DOMPurify.sanitize(html);
    } else {
        messageContent.textContent = content;
    }
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatContainer.appendChild(messageDiv);
    
    // 滚动到底部
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 添加加载指示器
function addLoadingIndicator() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant';
    messageDiv.id = 'loading-indicator';
    
    const avatar = createAvatar('assistant');
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'typing-indicator';
    typingDiv.innerHTML = '<span></span><span></span><span></span>';
    
    messageContent.appendChild(typingDiv);
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return 'loading-indicator';
}

// 移除加载指示器
function removeLoadingIndicator(id) {
    const indicator = document.getElementById(id);
    if (indicator) {
        indicator.remove();
    }
}

// 处理新对话
async function handleNewChat() {
    // 清除当前聊天显示
    chatContainer.innerHTML = `
        <div class="welcome-message">
            <img src="/static/img/logo.svg" alt="BrainBeacon Logo" class="welcome-logo">
            <h1>BrainBeacon · 大脑启智</h1>
            <p>我是 BrainBeacon 智能助理，可以帮您：</p>
            <ul>
                <li>分析空间虚拟扰动结果</li>
                <li>查询转录因子共调控关系</li>
                <li>访问 Open Targets 数据库获取基因功能与疾病信息等</li>
            </ul>
            <p>请开始提问吧！</p>
        </div>
    `;
    
    // 创建新会话
    try {
        const response = await fetch('/api/new_session', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        const data = await response.json();
        currentSessionId = data.session_id;
    } catch (error) {
        console.error('创建新会话失败:', error);
    }
}

