/**
 * Phonemizer module for Kokoro TTS
 * Adapted from kokoro.js/src/phonemize.js to work in web workers
 */

/**
 * Helper function to split a string on a regex, but keep the delimiters.
 * @param {string} text The text to split.
 * @param {RegExp} regex The regex to split on.
 * @returns {{match: boolean; text: string}[]} The split string.
 */
function split(text, regex) {
  const result = [];
  let prev = 0;
  for (const match of text.matchAll(regex)) {
    const fullMatch = match[0];
    if (prev < match.index) {
      result.push({ match: false, text: text.slice(prev, match.index) });
    }
    if (fullMatch.length > 0) {
      result.push({ match: true, text: fullMatch });
    }
    prev = match.index + fullMatch.length;
  }
  if (prev < text.length) {
    result.push({ match: false, text: text.slice(prev) });
  }
  return result;
}

/**
 * Helper function to split numbers into phonetic equivalents
 * @param {string} match The matched number
 * @returns {string} The phonetic equivalent
 */
function split_num(match) {
  if (match.includes(".")) {
    return match;
  } else if (match.includes(":")) {
    let [h, m] = match.split(":").map(Number);
    if (m === 0) {
      return `${h} o'clock`;
    } else if (m < 10) {
      return `${h} oh ${m}`;
    }
    return `${h} ${m}`;
  }
  let year = parseInt(match.slice(0, 4), 10);
  if (year < 1100 || year % 1000 < 10) {
    return match;
  }
  let left = match.slice(0, 2);
  let right = parseInt(match.slice(2, 4), 10);
  let suffix = match.endsWith("s") ? "s" : "";
  if (year % 1000 >= 100 && year % 1000 <= 999) {
    if (right === 0) {
      return `${left} hundred${suffix}`;
    } else if (right < 10) {
      return `${left} oh ${right}${suffix}`;
    }
  }
  return `${left} ${right}${suffix}`;
}

/**
 * Helper function to format monetary values
 * @param {string} match The matched currency
 * @returns {string} The formatted currency
 */
function flip_money(match) {
  const bill = match[0] === "$" ? "dollar" : "pound";
  if (isNaN(Number(match.slice(1)))) {
    return `${match.slice(1)} ${bill}s`;
  } else if (!match.includes(".")) {
    let suffix = match.slice(1) === "1" ? "" : "s";
    return `${match.slice(1)} ${bill}${suffix}`;
  }
  const [b, c] = match.slice(1).split(".");
  const d = parseInt(c.padEnd(2, "0"), 10);
  let coins = match[0] === "$" ? (d === 1 ? "cent" : "cents") : d === 1 ? "penny" : "pence";
  return `${b} ${bill}${b === "1" ? "" : "s"} and ${d} ${coins}`;
}

/**
 * Helper function to process decimal numbers
 * @param {string} match The matched number
 * @returns {string} The formatted number
 */
function point_num(match) {
  let [a, b] = match.split(".");
  return `${a} point ${b.split("").join(" ")}`;
}

/**
 * Normalize text for phonemization
 * @param {string} text The text to normalize
 * @returns {string} The normalized text
 */
function normalize_text(text) {
  return (
    text
      // 1. Handle quotes and brackets
      .replace(/['']/g, "'")
      .replace(/«/g, '"')
      .replace(/»/g, '"')
      .replace(/[""]/g, '"')
      .replace(/\(/g, "«")
      .replace(/\)/g, "»")

      // 2. Replace uncommon punctuation marks
      .replace(/、/g, ", ")
      .replace(/。/g, ". ")
      .replace(/！/g, "! ")
      .replace(/，/g, ", ")
      .replace(/：/g, ": ")
      .replace(/；/g, "; ")
      .replace(/？/g, "? ")

      // 3. Whitespace normalization
      .replace(/[^\S \n]/g, " ")
      .replace(/  +/, " ")
      .replace(/(?<=\n) +(?=\n)/g, "")

      // 4. Abbreviations
      .replace(/\bD[Rr]\.(?= [A-Z])/g, "Doctor")
      .replace(/\b(?:Mr\.|MR\.(?= [A-Z]))/g, "Mister")
      .replace(/\b(?:Ms\.|MS\.(?= [A-Z]))/g, "Miss")
      .replace(/\b(?:Mrs\.|MRS\.(?= [A-Z]))/g, "Mrs")
      .replace(/\betc\.(?! [A-Z])/gi, "etc")

      // 5. Normalize casual words
      .replace(/\b(y)eah?\b/gi, "$1e'a")

      // 5. Handle numbers and currencies
      .replace(/\d*\.\d+|\b\d{4}s?\b|(?<!:)\b(?:[1-9]|1[0-2]):[0-5]\d\b(?!:)/g, split_num)
      .replace(/(?<=\d),(?=\d)/g, "")
      .replace(/[$£]\d+(?:\.\d+)?(?: hundred| thousand| (?:[bm]|tr)illion)*\b|[$£]\d+\.\d\d?\b/gi, flip_money)
      .replace(/\d*\.\d+/g, point_num)
      .replace(/(?<=\d)-(?=\d)/g, " to ")
      .replace(/(?<=\d)S/g, " S")

      // 6. Handle possessives
      .replace(/(?<=[BCDFGHJ-NP-TV-Z])'?s\b/g, "'S")
      .replace(/(?<=X')S\b/g, "s")

      // 7. Handle hyphenated words/letters
      .replace(/(?:[A-Za-z]\.){2,} [a-z]/g, (m) => m.replace(/\./g, "-"))
      .replace(/(?<=[A-Z])\.(?=[A-Z])/gi, "-")

      // 8. Strip leading and trailing whitespace
      .trim()
  );
}

/**
 * Escapes regular expression special characters from a string by replacing them with their escaped counterparts.
 * @param {string} string The string to escape.
 * @returns {string} The escaped string.
 */
function escapeRegExp(string) {
  return string.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

const PUNCTUATION = ';:,.!?¡¿—…"«»""(){}[]';
const PUNCTUATION_PATTERN = new RegExp(`(\\s*[${escapeRegExp(PUNCTUATION)}]+\\s*)+`, "g");

/**
 * Simple phonemizer fallback when eSpeak-NG is not available
 * This is a basic implementation for web workers
 * @param {string} text The text to phonemize
 * @param {string} language The language to use
 * @returns {string} The phonemized text
 */
function simplePhonemizerFallback(text, language = "en-us") {
  // Basic phoneme mapping for common English words
  const phonemeMap = {
    'hello': 'h ə l oʊ',
    'hi': 'h aɪ',
    'this': 'ð ɪ s',
    'is': 'ɪ z',
    'a': 'ə',
    'test': 't ɛ s t',
    'of': 'ʌ v',
    'the': 'ð ə',
    'and': 'æ n d',
    'to': 't u',
    'for': 'f ɔr',
    'you': 'j u',
    'that': 'ð æ t',
    'with': 'w ɪ θ',
    'have': 'h æ v',
    'will': 'w ɪ l',
    'can': 'k æ n',
    'are': 'ɑr',
    'not': 'n ɑ t',
    'but': 'b ʌ t',
    'what': 'w ʌ t',
    'who': 'h u',
    'when': 'w ɛ n',
    'where': 'w ɛr',
    'why': 'w aɪ',
    'how': 'h aʊ',
    'yes': 'j ɛ s',
    'no': 'n oʊ',
    'please': 'p l i z',
    'thank': 'θ æ ŋ k',
    'thanks': 'θ æ ŋ k s',
    'good': 'ɡ ʊ d',
    'great': 'ɡ r eɪ t',
    'nice': 'n aɪ s',
    'well': 'w ɛ l',
    'okay': 'oʊ k eɪ',
    'ok': 'oʊ k eɪ',
    'right': 'r aɪ t',
    'now': 'n aʊ',
    'here': 'h ɪr',
    'there': 'ð ɛr',
    'time': 't aɪ m',
    'day': 'd eɪ',
    'today': 't ə d eɪ',
    'tomorrow': 't ə m ɔr oʊ',
    'yesterday': 'j ɛ s t ər d eɪ',
    'morning': 'm ɔr n ɪ ŋ',
    'afternoon': 'æ f t ər n u n',
    'evening': 'i v n ɪ ŋ',
    'night': 'n aɪ t',
    'work': 'w ɜr k',
    'home': 'h oʊ m',
    'house': 'h aʊ s',
    'place': 'p l eɪ s',
    'world': 'w ɜr l d',
    'country': 'k ʌ n t r i',
    'city': 's ɪ t i',
    'people': 'p i p ə l',
    'person': 'p ɜr s ə n',
    'man': 'm æ n',
    'woman': 'w ʊ m ə n',
    'child': 'tʃ aɪ l d',
    'children': 'tʃ ɪ l d r ə n',
    'friend': 'f r ɛ n d',
    'family': 'f æ m ə l i',
    'love': 'l ʌ v',
    'like': 'l aɪ k',
    'want': 'w ɑ n t',
    'need': 'n i d',
    'know': 'n oʊ',
    'think': 'θ ɪ ŋ k',
    'feel': 'f i l',
    'see': 's i',
    'hear': 'h ɪr',
    'say': 's eɪ',
    'tell': 't ɛ l',
    'talk': 't ɔ k',
    'speak': 's p i k',
    'listen': 'l ɪ s ə n',
    'understand': 'ʌ n d ər s t æ n d',
    'help': 'h ɛ l p',
    'try': 't r aɪ',
    'use': 'j u z',
    'make': 'm eɪ k',
    'do': 'd u',
    'get': 'ɡ ɛ t',
    'give': 'ɡ ɪ v',
    'take': 't eɪ k',
    'go': 'ɡ oʊ',
    'come': 'k ʌ m',
    'back': 'b æ k',
    'up': 'ʌ p',
    'down': 'd aʊ n',
    'out': 'aʊ t',
    'in': 'ɪ n',
    'on': 'ɑ n',
    'off': 'ɔ f',
    'over': 'oʊ v ər',
    'under': 'ʌ n d ər',
    'through': 'θ r u',
    'around': 'ə r aʊ n d',
    'between': 'b ɪ t w i n',
    'before': 'b ɪ f ɔr',
    'after': 'æ f t ər',
    'about': 'ə b aʊ t',
    'some': 's ʌ m',
    'many': 'm ɛ n i',
    'much': 'm ʌ tʃ',
    'more': 'm ɔr',
    'most': 'm oʊ s t',
    'other': 'ʌ ð ər',
    'another': 'ə n ʌ ð ər',
    'same': 's eɪ m',
    'different': 'd ɪ f ər ə n t',
    'new': 'n u',
    'old': 'oʊ l d',
    'big': 'b ɪ ɡ',
    'small': 's m ɔ l',
    'large': 'l ɑr dʒ',
    'little': 'l ɪ t ə l',
    'long': 'l ɔ ŋ',
    'short': 'ʃ ɔr t',
    'high': 'h aɪ',
    'low': 'l oʊ',
    'fast': 'f æ s t',
    'slow': 's l oʊ',
    'easy': 'i z i',
    'hard': 'h ɑr d',
    'difficult': 'd ɪ f ɪ k ə l t',
    'important': 'ɪ m p ɔr t ə n t',
    'interesting': 'ɪ n t r ə s t ɪ ŋ',
    'beautiful': 'b j u t ə f ə l',
    'happy': 'h æ p i',
    'sad': 's æ d',
    'angry': 'æ ŋ ɡ r i',
    'worried': 'w ɜr i d',
    'excited': 'ɪ k s aɪ t ɪ d',
    'tired': 't aɪ ər d',
    'hungry': 'h ʌ ŋ ɡ r i',
    'thirsty': 'θ ɜr s t i',
    'hot': 'h ɑ t',
    'cold': 'k oʊ l d',
    'warm': 'w ɔr m',
    'cool': 'k u l',
    'water': 'w ɔ t ər',
    'food': 'f u d',
    'eat': 'i t',
    'drink': 'd r ɪ ŋ k',
    'sleep': 's l i p',
    'wake': 'w eɪ k',
    'run': 'r ʌ n',
    'walk': 'w ɔ k',
    'sit': 's ɪ t',
    'stand': 's t æ n d',
    'lie': 'l aɪ',
    'stop': 's t ɑ p',
    'start': 's t ɑr t',
    'begin': 'b ɪ ɡ ɪ n',
    'end': 'ɛ n d',
    'finish': 'f ɪ n ɪ ʃ',
    'continue': 'k ə n t ɪ n j u',
    'wait': 'w eɪ t',
    'stay': 's t eɪ',
    'leave': 'l i v',
    'return': 'r ɪ t ɜr n',
    'open': 'oʊ p ə n',
    'close': 'k l oʊ s',
    'turn': 't ɜr n',
    'play': 'p l eɪ',
    'music': 'm j u z ɪ k',
    'song': 's ɔ ŋ',
    'movie': 'm u v i',
    'book': 'b ʊ k',
    'read': 'r i d',
    'write': 'r aɪ t',
    'learn': 'l ɜr n',
    'teach': 't i tʃ',
    'study': 's t ʌ d i',
    'school': 's k u l',
    'university': 'j u n ə v ɜr s ə t i',
    'college': 'k ɑ l ɪ dʒ',
    'student': 's t u d ə n t',
    'teacher': 't i tʃ ər',
    'doctor': 'd ɑ k t ər',
    'hospital': 'h ɑ s p ɪ t ə l',
    'phone': 'f oʊ n',
    'call': 'k ɔ l',
    'computer': 'k ə m p j u t ər',
    'internet': 'ɪ n t ər n ɛ t',
    'email': 'i m eɪ l',
    'message': 'm ɛ s ɪ dʒ',
    'information': 'ɪ n f ər m eɪ ʃ ə n',
    'question': 'k w ɛ s tʃ ə n',
    'answer': 'æ n s ər',
    'problem': 'p r ɑ b l ə m',
    'solution': 's ə l u ʃ ə n',
    'idea': 'aɪ d i ə',
    'plan': 'p l æ n',
    'project': 'p r ɑ dʒ ɛ k t',
    'business': 'b ɪ z n ə s',
    'company': 'k ʌ m p ə n i',
    'office': 'ɔ f ɪ s',
    'meeting': 'm i t ɪ ŋ',
    'team': 't i m',
    'group': 'ɡ r u p',
    'member': 'm ɛ m b ər',
    'leader': 'l i d ər',
    'manager': 'm æ n ɪ dʒ ər',
    'customer': 'k ʌ s t ə m ər',
    'service': 's ɜr v ɪ s',
    'product': 'p r ɑ d ʌ k t',
    'quality': 'k w ɑ l ə t i',
    'price': 'p r aɪ s',
    'money': 'm ʌ n i',
    'cost': 'k ɔ s t',
    'pay': 'p eɪ',
    'buy': 'b aɪ',
    'sell': 's ɛ l',
    'market': 'm ɑr k ɪ t',
    'store': 's t ɔr',
    'shop': 'ʃ ɑ p',
    'car': 'k ɑr',
    'drive': 'd r aɪ v',
    'travel': 't r æ v ə l',
    'trip': 't r ɪ p',
    'vacation': 'v eɪ k eɪ ʃ ə n',
    'holiday': 'h ɑ l ə d eɪ',
    'hotel': 'h oʊ t ɛ l',
    'restaurant': 'r ɛ s t ər ə n t',
    'party': 'p ɑr t i',
    'celebration': 's ɛ l ə b r eɪ ʃ ə n',
    'birthday': 'b ɜr θ d eɪ',
    'gift': 'ɡ ɪ f t',
    'present': 'p r ɛ z ə n t',
    'surprise': 's ər p r aɪ z'
  };

  // Convert to lowercase and split into words
  const words = text.toLowerCase().replace(/[^\w\s]/g, '').split(/\s+/);
  
  // Convert each word to phonemes
  const phonemes = words.map(word => {
    if (phonemeMap[word]) {
      return phonemeMap[word];
    } else {
      // Basic fallback: try to phonetically spell it out
      return word.split('').map(char => {
        const charMap = {
          'a': 'eɪ', 'b': 'b', 'c': 'k', 'd': 'd', 'e': 'i',
          'f': 'f', 'g': 'ɡ', 'h': 'h', 'i': 'aɪ', 'j': 'dʒ',
          'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'oʊ',
          'p': 'p', 'q': 'k w', 'r': 'r', 's': 's', 't': 't',
          'u': 'u', 'v': 'v', 'w': 'w', 'x': 'k s', 'y': 'waɪ',
          'z': 'z'
        };
        return charMap[char] || char;
      }).join(' ');
    }
  });
  
  return phonemes.join(' ');
}

/**
 * Phonemize text using either eSpeak-NG or a fallback
 * @param {string} text The text to phonemize
 * @param {"a"|"b"} language The language to use
 * @param {boolean} norm Whether to normalize the text
 * @returns {Promise<string>} The phonemized text
 */
export async function phonemize(text, language = "a", norm = true) {
  // 1. Normalize text
  if (norm) {
    text = normalize_text(text);
  }

  // 2. Try to use eSpeak-NG phonemizer if available
  let phonemizedText = '';
  
  try {
    // Try to import the phonemizer
    const { phonemize: espeakng } = await import("phonemizer");
    
    // Split into chunks to preserve punctuation
    const sections = split(text, PUNCTUATION_PATTERN);
    
    // Convert each section to phonemes
    const lang = language === "a" ? "en-us" : "en";
    const ps = (await Promise.all(sections.map(async ({ match, text }) => 
      match ? text : (await espeakng(text, lang)).join(" ")
    ))).join("");
    
    phonemizedText = ps;
    
  } catch (error) {
    console.warn('eSpeak-NG phonemizer not available, using fallback:', error.message);
    
    // Use simple fallback phonemizer
    phonemizedText = simplePhonemizerFallback(text, language === "a" ? "en-us" : "en");
  }

  // 3. Post-process phonemes
  let processed = phonemizedText
    // https://en.wiktionary.org/wiki/kokoro#English
    .replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
    .replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹəʊ")
    .replace(/ʲ/g, "j")
    .replace(/r/g, "ɹ")
    .replace(/x/g, "k")
    .replace(/ɬ/g, "l")
    .replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
    .replace(/ z(?=[;:,.!?¡¿—…"«»"" ]|$)/g, "z");

  // 4. Additional post-processing for American English
  if (language === "a") {
    processed = processed.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
  }
  
  return processed.trim();
}

/**
 * Quick phonemize for simple cases
 * @param {string} text The text to phonemize
 * @returns {string} The phonemized text
 */
export function quickPhonemize(text) {
  return simplePhonemizerFallback(normalize_text(text));
}

// Export all functions for use in workers
export { normalize_text, simplePhonemizerFallback, split, split_num, flip_money, point_num };
