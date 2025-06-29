import prompts from './prompts.js'
import { llmComplete, llmLogits } from './llm.js'
import { deriveVAD, getEmotionsList } from './emotions.js'

export async function evaluateAvatarEmotion({ ctx, thread }){
	return await queryEmotions({
		ctx,
		thread,
		prompt: prompts.evaluation.emotion.prompt,
		preface: prompts.evaluation.emotion.preface
	})
}

export async function evaluateAvatarAffect({ ctx, thread }){

}

export async function evaluateAvatarEsteem({ ctx, thread }){
	
}

export async function evaluateAvatarGoal({ ctx, thread }){
	let resultThread = await llmComplete({
		ctx,
		thread,
		prompt: prompts.evaluation.goal.prompt,
		preface: prompts.evaluation.goal.preface,
		mode: 'phrase'
	})

	return resultThread.last.withoutPreface
}

export async function evaluateUserEmotion({ ctx, thread }){
	
}

async function queryEmotions({ ctx, ...query }){
	let logits = await llmLogits({
		ctx,
		...query,
		probe: getEmotionsList()
	})

	return {
		logits,
		vad: deriveVAD({ emotionLogits: logits })
	}
}