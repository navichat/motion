# Conversation Components

Purpose
- Interfaces and helpers to connect avatar state/VRM with conversation flows.

Key Modules
- VRMConversationInterface.js — attach conversation signals to avatar

Usage
```js
// const iface = new VRMConversationInterface(vrm, { tts, stt });
// iface.on('expression', e => {/* update VRM */});
```

Related Tests
- dev/web_viewer/src/testing/integration/
