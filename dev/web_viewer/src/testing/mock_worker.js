self.onmessage = async (event) => {
    if (event.data.type === 'audio') {
        // Simulate VAD and Whisper processing by immediately sending a transcription
        self.postMessage({ type: 'output', text: 'This is a test sentence.', result: new Float32Array(100) });
    }
};
