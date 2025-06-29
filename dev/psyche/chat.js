import log from '@mwni/log'
import prompts from './prompts.js'
import { LLMThread, llmComplete } from './llm.js'
import { evaluateAvatarAffect, evaluateAvatarEmotion, evaluateAvatarEsteem, evaluateAvatarGoal, evaluateUserEmotion } from './evaluation.js'


export async function getNextResponse({ ctx, meeting }){
	return await new Promise((resolve, reject) => {
		let promise
		let emitResponse = response => {
			meeting.messages.push({
				role: 'avatar',
				...response
			})
			resolve(response)
		}

		if(meeting.messages.length === 0){
			promise = firstChatRound({ ctx, meeting, emitResponse })
		}else{
			promise = nextChatRound({ ctx, meeting, emitResponse })
		}

		promise.catch(error => {
			log.warn(`error while generating response:`, error)
			reject(error)
		})
	})
}

async function firstChatRound({ ctx, meeting, emitResponse }){
	meeting.state = {
		character: prompts.genesis.character,
		scenario: prompts.genesis.scenario,
		mood: prompts.genesis.mood,
		baseGoal: prompts.genesis.baseGoal,
	}

	emitResponse({
		text: prompts.genesis.greeting,
		tone: 'With kindness, calmly'
	})
}

async function nextChatRound({ ctx, meeting, emitResponse }){
	ctx = { ...ctx, log: meeting.log }

	let opener = meeting.messages[0].text
	let transcript = composeTranscript(meeting.messages.slice(1, -1))
	let lastUserMessage = meeting.messages.slice(-1)[0].text

	let thread = new LLMThread(
		{ system: meeting.state.character },
		{ user: prompts.setup.scenario.prompt },
		{ assistant: prompts.setup.scenario.answer },
		{ user: prompts.setup.recap.prompt },
		{ assistant: prompts.setup.recap.answer },
		{ user: prompts.setup.state.prompt },
		{ assistant: prompts.setup.state.answer }
	).fill({
		...meeting.state,
		opener,
		transcript,
		lastUserMessage
	})

	let [
		avatarEmotion,
		avatarGoal,
		avatarAffect,
		avatarEsteem,
		userEmotion,
	] = await Promise.all([
		evaluateAvatarEmotion({ ctx, thread }),
		evaluateAvatarGoal({ ctx, thread }),
		evaluateAvatarAffect({ ctx, thread }),
		evaluateAvatarEsteem({ ctx, thread }),
		evaluateUserEmotion({ ctx, thread }),
	])

	console.log('avatar emotion:', avatarEmotion)
	console.log('avatar goal:', avatarGoal)

	let responseThread = await llmComplete({
		ctx,
		thread,
		prompt: prompts.response.prompt.format({ goal: avatarGoal }),
		preface: prompts.response.preface,
		mode: 'quote'
	})
	let response = responseThread.last.withoutPreface.slice(0, -1)

	emitResponse({
		text: response
	})
}

function composeTranscript(messages){
	return messages
		.map(({ role, text }) => prompts.transcript[role].format({ text }))
		.join('\n')
}