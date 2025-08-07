#!/usr/bin/env node

// Test script for the phonemizer module
// Usage: node test_phonemizer.js

import { phonemize, quickPhonemize, normalize_text } from './phonemizer.js';

console.log('🎵 Phonemizer Test Script');
console.log('========================\n');

// Test texts
const testTexts = [
    "Hello world!",
    "This is a test of the phonemizer with numbers like 123 and $45.67.",
    "I have 3 cats and 2 dogs.",
    "The price is $99.99 for the premium version.",
    "Call me at 555-1234 or email test@example.com.",
    "Dr. Smith will see you at 3:30 PM on Monday, January 15th.",
    "The temperature is 72.5°F today.",
    "我爱你 (I love you in Chinese)", // Test non-ASCII
    "Café, naïve, résumé", // Test accented characters
    "U.S.A. vs. U.K. comparison"
];

async function testPhonemizerWithText(text) {
    console.log(`\n📝 Testing: "${text}"`);
    console.log('─'.repeat(60));
    
    try {
        // Test normalization
        const normalized = normalize_text(text);
        console.log(`Normalized: "${normalized}"`);
        
        // Test quick phonemization (fallback)
        const quickPhonemes = quickPhonemize(normalized);
        console.log(`Quick phonemes: "${quickPhonemes}"`);
        
        // Test full phonemization (with eSpeak-NG if available)
        const fullPhonemes = await phonemize(normalized);
        console.log(`Full phonemes: "${fullPhonemes}"`);
        
        if (quickPhonemes !== fullPhonemes) {
            console.log('✅ eSpeak-NG detected and used');
        } else {
            console.log('⚠️  Using fallback phonemization');
        }
        
    } catch (error) {
        console.error(`❌ Error: ${error.message}`);
    }
}

async function runAllTests() {
    console.log('Starting comprehensive phonemizer tests...\n');
    
    for (const text of testTexts) {
        await testPhonemizerWithText(text);
    }
    
    console.log('\n🎉 All tests completed!');
    console.log('\nNote: If you see "Using fallback phonemization" for all tests,');
    console.log('it means eSpeak-NG is not available in this environment.');
    console.log('This is expected for browser/worker environments.');
}

// Run the tests
runAllTests().catch(console.error);
