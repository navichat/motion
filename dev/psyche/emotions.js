export const vadMap = {
	happy: { 		valence: 1, 	arousal: 0.5, 	dominance: 0 	},
	joyful: { 		valence: 1, 	arousal: 0.7, 	dominance: 0.2 	},
	excited: { 		valence: 0.5, 	arousal: 0.9, 	dominance: 0.5 	},
	horny: { 		valence: 0.5, 	arousal: 1, 	dominance: 0.3	},
	statisfied: { 	valence: 0.8, 	arousal: 0, 	dominance: 0.8 	},
	pleased: { 		valence: 0.9, 	arousal: 0.3, 	dominance: 0.5 	},
	amused: { 		valence: 0.7, 	arousal: 0.3, 	dominance: 0.8 	},
	calm: { 		valence: 0.4, 	arousal: -0.7, 	dominance: 0 	},
	bored: { 		valence: -0.5, 	arousal: -0.9, 	dominance: -0.2	},
	astonished: { 	valence: 0.2, 	arousal: 0.8, 	dominance: 0.2 	},
	scared: { 		valence: -1, 	arousal: 0.8, 	dominance: -1 	},
	shocked: { 		valence: -0.5, 	arousal: 1, 	dominance: -0.5 },
	distressed: { 	valence: -0.7, 	arousal: -0.3, 	dominance: -0.7	},
	alarmed: { 		valence: -0.3, 	arousal: 0.9, 	dominance: 0 	},
	angry: { 		valence: -0.7, 	arousal: 1, 	dominance: 0.7 	},
	annoyed: { 		valence: -0.6, 	arousal: 0.6, 	dominance: -0.2 },
	frustrated: { 	valence: -0.8, 	arousal: -0.3, 	dominance: -0.5	},
	miserable: { 	valence: -1, 	arousal: -0.7, 	dominance: -1 	},
	awful: { 		valence: -1, 	arousal: 0, 	dominance: 0 	},
	sad: { 			valence: -0.8, 	arousal: -0.5, 	dominance: -0.7 },
	depressed: { 	valence: -1, 	arousal: -1, 	dominance: -1 	},
	disgusted: { 	valence: -1, 	arousal: 0.5, 	dominance: 1 	},
	ashamed: { 		valence: -0.5, 	arousal: -0.5, 	dominance: -1 	},
}

export function getEmotionsList(){
	return Object.keys(vadMap)
}

export function deriveVAD({ emotionLogits }){
	let emotions = softmax(emotionLogits)
	let vad = { valence: 0, arousal: 0, dominance: 0 }
	let weight = 0

	for(let [emotion, probability] of Object.entries(emotions)){
		for(let key of Object.keys(vad)){
			vad[key] += vadMap[emotion][key] * probability
			weight += probability
		}
	}

	for(let key of Object.keys(vad)){
		vad[key] *= 3 / weight
	}

	return vad
}

export function softmax(logits) {
	const entries = Object.entries(logits)
	const maxLogit = Math.max(...entries.map(([_, value]) => value))
	const scores = entries.map(([key, value]) => [key, Math.exp(value - maxLogit)])
	const sum = scores.reduce((acc, [_, value]) => acc + value, 0)

	return Object.fromEntries(scores.map(([key, value]) => [key, value / sum]))
}