#!/usr/bin/env bash

# Quick status check for the conversation worker and TTS integration

echo "🔍 Checking Conversation Worker and TTS Integration Status"
echo "======================================================="

# Check if core files exist
echo ""
echo "📁 Core Files:"
echo "  conversationWorkerWorking.js: $(test -f js/conversationWorkerWorking.js && echo '✅ EXISTS' || echo '❌ MISSING')"
echo "  phonemizer.js: $(test -f js/phonemizer.js && echo '✅ EXISTS' || echo '❌ MISSING')"
echo "  chat-tts-integration.js: $(test -f js/chat-tts-integration.js && echo '✅ EXISTS' || echo '❌ MISSING')"

# Check test files
echo ""
echo "🧪 Test Files:"
echo "  test_enhanced_conversation_worker.html: $(test -f test_enhanced_conversation_worker.html && echo '✅ EXISTS' || echo '❌ MISSING')"
echo "  test_phonemizer.html: $(test -f test_phonemizer.html && echo '✅ EXISTS' || echo '❌ MISSING')"
echo "  test_phonemizer.js: $(test -f test_phonemizer.js && echo '✅ EXISTS' || echo '❌ MISSING')"

# Check for key integration points
echo ""
echo "🔗 Integration Points:"
echo "  Phonemizer import in worker: $(grep -q "import.*phonemizer" js/conversationWorkerWorking.js && echo '✅ INTEGRATED' || echo '❌ MISSING')"
echo "  Kokoro TTS wrapper: $(grep -q "class.*Kokoro" js/conversationWorkerWorking.js && echo '✅ FOUND' || echo '❌ MISSING')"
echo "  Metadata handling: $(grep -q "metadata.*phonemes" js/conversationWorkerWorking.js && echo '✅ ENHANCED' || echo '❌ BASIC')"

# Check voice files (if available)
echo ""
echo "🎵 Voice Files:"
VOICE_COUNT=$(find . -name "*.bin" 2>/dev/null | wc -l)
echo "  Available voice files: $VOICE_COUNT"

if [ $VOICE_COUNT -gt 0 ]; then
    echo "  Voice files found:"
    find . -name "*.bin" 2>/dev/null | head -5 | sed 's/^/    /'
    if [ $VOICE_COUNT -gt 5 ]; then
        echo "    ... and $((VOICE_COUNT - 5)) more"
    fi
fi

echo ""
echo "🚀 Quick Start Commands:"
echo "  1. Test phonemizer: node test_phonemizer.js"
echo "  2. Test in browser: open test_phonemizer.html"
echo "  3. Test conversation worker: open test_enhanced_conversation_worker.html"
echo "  4. Start local server: python -m http.server 8000"

echo ""
echo "✅ Status check complete!"
