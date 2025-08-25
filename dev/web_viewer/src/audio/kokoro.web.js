/**
 * Kokoro TTS Local Fallback (browser-only, no speechSynthesis)
 * Generates deterministic PCM offline so headless/CI never hangs.
 */
const KokoroTTS = {
    async from_pretrained(modelPath, options = {}) {
        console.warn('🔄 Kokoro TTS: using local offline PCM fallback (model: ' + (modelPath||'fallback') + ')');
        return {
            async generate(text, voice = 'default', opts = {}) {
                const sampleRate = 22050;
                const lenText = (text || '').length;
                const duration = Math.min(10, Math.max(0.6, lenText * 0.06));
                const n = Math.floor(sampleRate * duration);
                const pcm = new Float32Array(n);
                // Simple quasi-voice: base tone + slow AM + slight freq wobble
                const f0 = 210;
                for (let i = 0; i < n; i++) {
                    const t = i / sampleRate;
                    const wob = 2 + 1.5 * Math.sin(2 * Math.PI * 0.5 * t);
                    pcm[i] = 0.08 * Math.sin(2 * Math.PI * (f0 + wob) * t) * (0.6 + 0.4 * Math.sin(2 * Math.PI * 3 * t));
                }
                return pcm; // Float32Array @ 22050 Hz
            },
            async generateWithTimings(text, voice = 'default') {
                const audio = await this.generate(text, voice);
                return { audio, phonemes: [], timings: [] };
            },
            getVoices() { return [{ name: 'default', lang: 'en-US', gender: 'neutral' }]; }
        };
    },
    async quickSynthesize(text, options = {}) {
        const model = await this.from_pretrained('fallback', options);
        return await model.generate(text, options.voice);
    }
};

if (typeof window !== 'undefined') { window.KokoroTTS = KokoroTTS; }
console.log('Kokoro TTS local offline fallback loaded');
