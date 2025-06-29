# Chat Interface Project Documentation

## Overview

The Chat Interface project is a comprehensive web-based application that provides real-time character animation and interactive chat capabilities. It combines modern web technologies with 3D character rendering to create immersive conversational experiences with animated avatars.

## Project Structure

```
chat/
├── assets/                    # Static assets and media files
│   ├── animations/           # Character animation files
│   ├── avatars/             # 3D character models and textures
│   └── scenes/              # Environment and scene assets
├── psyche/                  # AI conversation engine
│   ├── package.json         # Psyche module dependencies
│   ├── index.js            # Main psyche export
│   ├── chat.js             # Chat logic and response generation
│   ├── emotions.js         # Emotion processing and management
│   ├── evaluation.js       # Response evaluation and scoring
│   ├── llm.js             # Large language model integration
│   ├── prompts.js          # Prompt management and templates
│   ├── prompts.yml         # YAML prompt configurations
│   └── session.js          # Session management and persistence
├── server/                  # Backend server components
│   ├── package.json        # Server dependencies
│   ├── server.js           # Main server application
│   ├── accounts.js         # User account management
│   ├── avatar.js           # Avatar generation and management
│   ├── caching.js          # Caching and performance optimization
│   ├── meeting.js          # Meeting and session coordination
│   ├── schema.json         # Database schema definition
│   └── session.js          # Server-side session handling
└── webapp/                  # Frontend web application
    ├── package.json        # Frontend dependencies
    ├── app.html           # Main application HTML
    ├── app.js             # Main application logic
    ├── app.scss           # Application styling
    ├── dev.js             # Development utilities
    ├── home.js            # Home page functionality
    ├── meeting.js         # Meeting interface
    ├── player.loader.js   # 3D player loading utilities
    ├── server.js          # Frontend server
    ├── assets/            # Frontend-specific assets
    └── ui/                # User interface components
```

## Architecture Overview

### Multi-tier Architecture
```
Frontend (webapp) ↔ Backend (server) ↔ AI Engine (psyche) ↔ Database (MySQL)
                 ↕                    ↕                   ↕
            3D Rendering         Avatar Management    Conversation State
            User Interface       Session Handling     Emotion Processing
            WebSocket Comm.      Caching System       LLM Integration
```

## Core Components

### 1. Psyche - AI Conversation Engine

#### Purpose
Advanced AI system for generating contextual, emotionally-aware responses in character conversations.

#### Key Features
- **Conversation Management:** Maintains context and flow
- **Emotion Processing:** Analyzes and responds to emotional cues
- **LLM Integration:** Connects to large language models for response generation
- **Prompt Engineering:** Sophisticated prompt templates and management
- **Session Persistence:** Maintains long-term conversation memory

#### Dependencies
```json
{
  "@mwni/events": "^3.0.0",      // Event handling system
  "@mwni/log": "^2.2.0",         // Logging utilities
  "@structdb/mysql": "^1.3.3-alpha", // Database connectivity
  "yaml": "^2.3.3"               // YAML configuration parsing
}
```

#### Key Files
- `chat.js` - Core conversation logic
- `emotions.js` - Emotion analysis and response
- `llm.js` - Language model integration
- `prompts.yml` - Conversation prompts and templates

### 2. Server - Backend Infrastructure

#### Purpose
Robust backend system handling user management, avatar generation, session coordination, and real-time communication.

#### Key Features
- **Account Management:** User registration, authentication, and profiles
- **Avatar Generation:** Dynamic character creation and customization
- **Session Coordination:** Multi-user meeting and chat session management
- **Caching System:** Performance optimization for frequent operations
- **CloudKit Integration:** Cloud-based computing resources
- **Database Management:** MySQL-based data persistence

#### Dependencies
```json
{
  "@cloudkit/client": "^1.3.1",     // Cloud computing integration
  "@mwni/events": "^3.0.0",         // Event system
  "@mwni/log": "^2.2.0",            // Logging
  "@mwni/random": "^1.0.1",         // Random utilities
  "@mwni/toml": "^1.0.0",           // TOML configuration
  "@navi/engine": "file:../../engine/core", // 3D engine core
  "@navichat/psyche": "file:../psyche",     // AI conversation engine
  "@structdb/mysql": "^1.3.3-alpha",       // Database layer
  "bcrypt": "^5.1.1"                       // Password hashing
}
```

#### Key Files
- `server.js` - Main server application and routing
- `accounts.js` - User account operations
- `avatar.js` - Character generation and management
- `meeting.js` - Session and meeting coordination
- `schema.json` - Database structure definition

### 3. WebApp - Frontend Interface

#### Purpose
Modern web interface providing immersive 3D character interaction with real-time communication capabilities.

#### Key Features
- **3D Character Rendering:** Real-time avatar animation and display
- **Real-time Communication:** WebSocket-based chat and interaction
- **Responsive Design:** Modern UI adapting to different screen sizes
- **Audio Integration:** Sound effects and character voice synthesis
- **Build System:** Modern JavaScript bundling and optimization

#### Dependencies
```json
{
  "@koa/router": "^12.0.0",           // Server routing
  "@mwni/events": "^3.0.0",           // Event handling
  "@mwni/fetch": "^1.0.0",            // HTTP client utilities
  "@mwni/log": "^2.2.0",              // Logging system
  "@mwni/socket": "^1.3.0",           // Socket communication
  "@mwni/wss": "^1.1.0",              // WebSocket server
  "@navi/player": "file:../../engine/player", // 3D player engine
  "animejs": "^3.2.1",                // Animation library
  "howler": "^2.2.4",                 // Audio management
  "koa": "^2.14.2",                   // Web framework
  "mithril": "^2.2.2"                 // Frontend framework
}
```

#### Development Tools
```json
{
  "chokidar": "^3.5.3",              // File watching
  "esbuild": "0.19.2",               // Fast JavaScript bundling
  "postcss": "^8.4.30",              // CSS processing
  "postcss-import": "^15.1.0",       // CSS import handling
  "postcss-nested": "^6.0.1"         // Nested CSS support
}
```

## Installation and Setup

### Prerequisites
- **Node.js:** 16+ required for all components
- **MySQL:** Database server for data persistence
- **Modern Browser:** Chrome, Firefox, Safari for WebGL support

### Installation Steps

#### 1. Install Dependencies
```bash
# Install psyche dependencies
cd chat/psyche
npm install

# Install server dependencies
cd ../server
npm install

# Install webapp dependencies
cd ../webapp
npm install
```

#### 2. Database Setup
```bash
# Create MySQL database
mysql -u root -p
CREATE DATABASE navichat;

# Import schema
mysql -u root -p navichat < server/schema.json
```

#### 3. Configuration
Create configuration files for each component:

**Server Configuration:**
```toml
# server/config.toml
[database]
host = "localhost"
user = "your_user"
password = "your_password"
database = "navichat"

[server]
port = 8080
computeEndpoint = "https://your-compute-endpoint.com"
cacheDir = "./cache"
resourcesDir = "./assets"
```

#### 4. Build and Start
```bash
# Build webapp
cd webapp
npm run build

# Start server
cd ../server
node server.js

# Start webapp (in development)
cd ../webapp
node server.js
```

## Usage

### Basic Chat Interface

#### Starting a Session
1. **Access Application:** Navigate to `http://localhost:3000`
2. **Create Account:** Register or login to access features
3. **Choose Avatar:** Select or customize character appearance
4. **Start Conversation:** Begin chatting with AI-powered characters

#### Real-time Features
- **Live Animation:** Characters respond with appropriate gestures and expressions
- **Voice Synthesis:** Optional text-to-speech for character responses
- **Emotion Recognition:** Characters adapt behavior based on conversation tone
- **Multi-user Support:** Join meetings with multiple participants

### Advanced Features

#### Avatar Customization
```javascript
// Customize character appearance
const avatarConfig = {
  appearance: {
    hairColor: '#8B4513',
    skinTone: 'medium',
    clothing: 'casual'
  },
  personality: {
    traits: ['friendly', 'helpful', 'enthusiastic'],
    mood: 'positive'
  },
  animations: {
    idle: 'casual_idle',
    speaking: 'expressive_talk',
    listening: 'attentive_nod'
  }
};
```

#### Meeting Management
```javascript
// Create and manage meetings
const meeting = {
  name: 'Team Standup',
  participants: ['user1', 'user2', 'ai_assistant'],
  settings: {
    allowScreenShare: true,
    recordSession: false,
    maxParticipants: 10
  }
};
```

## API Reference

### Server API Endpoints

#### Authentication
```http
POST /api/auth/login
POST /api/auth/register
POST /api/auth/logout
GET  /api/auth/profile
```

#### Avatar Management
```http
GET    /api/avatars          # List available avatars
POST   /api/avatars          # Create new avatar
PUT    /api/avatars/:id      # Update avatar
DELETE /api/avatars/:id      # Delete avatar
```

#### Session Management
```http
GET    /api/sessions         # List user sessions
POST   /api/sessions         # Create new session
GET    /api/sessions/:id     # Get session details
PUT    /api/sessions/:id     # Update session
DELETE /api/sessions/:id     # End session
```

#### Meeting Coordination
```http
GET    /api/meetings         # List meetings
POST   /api/meetings         # Create meeting
GET    /api/meetings/:id     # Get meeting details
POST   /api/meetings/:id/join # Join meeting
POST   /api/meetings/:id/leave # Leave meeting
```

### WebSocket Events

#### Client to Server
```javascript
// Join conversation
socket.send({
  type: 'join_conversation',
  conversationId: 'conv_123',
  userId: 'user_456'
});

// Send message
socket.send({
  type: 'send_message',
  message: 'Hello, how are you?',
  timestamp: Date.now()
});

// Update avatar state
socket.send({
  type: 'avatar_update',
  animation: 'wave',
  emotion: 'happy'
});
```

#### Server to Client
```javascript
// Message received
{
  type: 'message_received',
  sender: 'ai_character',
  message: 'I\'m doing great, thanks for asking!',
  emotion: 'cheerful',
  animation: 'smile'
}

// Avatar state change
{
  type: 'avatar_state_changed',
  userId: 'user_123',
  state: {
    position: [0, 0, 0],
    rotation: [0, 0.5, 0],
    animation: 'talking'
  }
}
```

## Development

### Frontend Development

#### Component Structure
```javascript
// ui/chat-component.js
export default {
  view: ({ attrs }) => {
    return m('.chat-interface', [
      m('.messages', attrs.messages.map(msg => 
        m('.message', {
          class: msg.sender === 'user' ? 'user-message' : 'ai-message'
        }, msg.content)
      )),
      m('.input-area', [
        m('input[type=text]', {
          value: attrs.inputValue,
          oninput: (e) => attrs.onInput(e.target.value),
          placeholder: 'Type your message...'
        }),
        m('button', {
          onclick: attrs.onSend
        }, 'Send')
      ])
    ]);
  }
};
```

#### 3D Rendering Integration
```javascript
// Integration with 3D player engine
import { Player } from '@navi/player';

const initializeCharacter = async (containerId, avatarConfig) => {
  const player = new Player({
    container: document.getElementById(containerId),
    assets: avatarConfig.assets,
    animations: avatarConfig.animations
  });
  
  await player.loadCharacter(avatarConfig.model);
  return player;
};
```

### Backend Development

#### Adding New API Endpoints
```javascript
// server/routes/custom.js
export const customRoutes = (router, ctx) => {
  router.get('/api/custom/data', async (koaCtx) => {
    const data = await ctx.db.query('SELECT * FROM custom_table');
    koaCtx.body = { success: true, data };
  });
  
  router.post('/api/custom/action', async (koaCtx) => {
    const { action, parameters } = koaCtx.request.body;
    const result = await processCustomAction(action, parameters);
    koaCtx.body = result;
  });
};
```

#### Database Operations
```javascript
// Using the database layer
const createUser = async (userData) => {
  const userId = await ctx.db.insert('users', {
    username: userData.username,
    email: userData.email,
    passwordHash: await bcrypt.hash(userData.password, 10),
    createdAt: new Date()
  });
  
  return userId;
};
```

### AI Integration Development

#### Custom Prompt Templates
```yaml
# psyche/prompts.yml
character_responses:
  friendly:
    greeting: |
      You are a friendly and enthusiastic character. 
      Respond to the user's greeting in a warm, welcoming manner.
      Keep responses conversational and engaging.
    
  professional:
    greeting: |
      You are a professional assistant character.
      Provide helpful, accurate information in a polite manner.
      Maintain professionalism while being approachable.
```

#### Emotion Processing
```javascript
// psyche/emotions.js
export const analyzeEmotion = (text) => {
  const emotionKeywords = {
    happy: ['great', 'awesome', 'excellent', 'wonderful'],
    sad: ['terrible', 'awful', 'disappointing', 'bad'],
    excited: ['amazing', 'incredible', 'fantastic', 'wow']
  };
  
  const detectedEmotion = detectEmotion(text, emotionKeywords);
  return {
    primary: detectedEmotion,
    confidence: calculateConfidence(text, detectedEmotion),
    intensity: measureIntensity(text)
  };
};
```

## Performance Optimization

### Frontend Optimization
- **Asset Loading:** Lazy loading for 3D models and textures
- **WebGL Optimization:** Efficient rendering pipeline for real-time animation
- **Bundle Splitting:** Code splitting for faster initial load times
- **Caching:** Service worker for offline capability

### Backend Optimization
- **Database Indexing:** Optimized queries for user and session data
- **Connection Pooling:** Efficient database connection management
- **Caching Layer:** Redis for session and frequently accessed data
- **Load Balancing:** Horizontal scaling for high traffic

### Real-time Communication
- **WebSocket Optimization:** Efficient message serialization
- **Compression:** Message compression for bandwidth efficiency
- **Connection Management:** Graceful handling of disconnections
- **Rate Limiting:** Protection against spam and abuse

## Security Considerations

### Authentication & Authorization
- **Secure Password Handling:** bcrypt for password hashing
- **Session Management:** Secure session tokens and expiration
- **Input Validation:** Comprehensive validation for all user inputs
- **Rate Limiting:** Protection against brute force attacks

### Data Protection
- **Database Security:** Parameterized queries to prevent SQL injection
- **XSS Prevention:** Content Security Policy and input sanitization
- **HTTPS Enforcement:** SSL/TLS for all communications
- **Privacy Controls:** User data management and deletion capabilities

## Deployment

### Production Setup

#### Docker Configuration
```dockerfile
# Dockerfile for server
FROM node:16-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 8080
CMD ["node", "server.js"]
```

#### Environment Configuration
```bash
# Production environment variables
NODE_ENV=production
DATABASE_URL=mysql://user:pass@host:port/db
REDIS_URL=redis://cache-host:6379
CLOUDKIT_API_KEY=your_api_key
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### Monitoring and Logging
- **Application Monitoring:** Health checks and performance metrics
- **Error Tracking:** Comprehensive error logging and alerting
- **User Analytics:** Usage patterns and feature adoption
- **Performance Monitoring:** Response times and resource utilization

## Testing

### Unit Testing
```javascript
// tests/psyche/chat.test.js
import { describe, it, expect } from '@jest/globals';
import { generateResponse } from '../chat.js';

describe('Chat Response Generation', () => {
  it('should generate appropriate response for greeting', async () => {
    const input = 'Hello there!';
    const response = await generateResponse(input, 'friendly');
    
    expect(response).toBeDefined();
    expect(response.emotion).toBe('positive');
    expect(response.text).toContain('hello');
  });
});
```

### Integration Testing
```javascript
// tests/server/api.test.js
import request from 'supertest';
import { app } from '../server.js';

describe('API Endpoints', () => {
  it('should create new user account', async () => {
    const userData = {
      username: 'testuser',
      email: 'test@example.com',
      password: 'securepassword'
    };
    
    const response = await request(app)
      .post('/api/auth/register')
      .send(userData)
      .expect(201);
    
    expect(response.body.success).toBe(true);
  });
});
```

### End-to-End Testing
```javascript
// tests/e2e/chat-flow.test.js
import { test, expect } from '@playwright/test';

test('complete chat interaction flow', async ({ page }) => {
  await page.goto('http://localhost:3000');
  
  // Login
  await page.fill('[data-testid=username]', 'testuser');
  await page.fill('[data-testid=password]', 'password');
  await page.click('[data-testid=login-button]');
  
  // Start chat
  await page.fill('[data-testid=message-input]', 'Hello!');
  await page.click('[data-testid=send-button]');
  
  // Verify response
  await expect(page.locator('[data-testid=ai-response]')).toBeVisible();
});
```

## Troubleshooting

### Common Issues

#### WebSocket Connection Problems
```javascript
// Debug WebSocket connectivity
const debugWebSocket = (url) => {
  const ws = new WebSocket(url);
  
  ws.onopen = () => console.log('WebSocket connected');
  ws.onerror = (error) => console.error('WebSocket error:', error);
  ws.onclose = (event) => console.log('WebSocket closed:', event.code);
  
  return ws;
};
```

#### 3D Rendering Issues
- **WebGL Support:** Check browser WebGL compatibility
- **GPU Memory:** Monitor GPU memory usage for large models
- **Performance:** Optimize model complexity for target devices
- **Compatibility:** Test across different browsers and devices

#### Database Connection Issues
```javascript
// Database connection debugging
const testDatabaseConnection = async () => {
  try {
    await ctx.db.query('SELECT 1');
    console.log('Database connection successful');
  } catch (error) {
    console.error('Database connection failed:', error);
  }
};
```

### Debug Tools
- **Browser DevTools:** Network, console, and performance analysis
- **Database Monitoring:** Query performance and connection status
- **Server Logs:** Comprehensive logging for backend operations
- **WebSocket Inspector:** Real-time message monitoring

## Future Enhancements

### Planned Features
- **Mobile App:** Native mobile applications for iOS and Android
- **VR Support:** Virtual reality integration for immersive experiences
- **Advanced AI:** More sophisticated conversation capabilities
- **Custom Characters:** User-generated character creation tools

### Technology Roadmap
- **Performance:** WebAssembly for computationally intensive operations
- **Rendering:** Advanced graphics techniques for realistic characters
- **AI Integration:** Latest language models and conversation AI
- **Scalability:** Microservices architecture for large-scale deployment

## Contributing

### Development Guidelines
- **Code Style:** Follow ESLint and Prettier configurations
- **Testing:** Maintain high test coverage for all components
- **Documentation:** Update documentation for all changes
- **Version Control:** Use conventional commit messages

### Pull Request Process
1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request with detailed description

## License

See individual project licenses for specific terms and conditions.

## Support and Community

- **GitHub Issues:** Bug reports and feature requests
- **Documentation:** Comprehensive guides and API reference
- **Community Forums:** Developer discussions and support
- **Discord/Slack:** Real-time community chat and support
