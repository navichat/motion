import log from '@mwni/log'
import { firstChatRound, nextChatRound } from './chat.js'
import { models as llmModels } from './llm.js'

export async function createSession({ ctx, account, persona }){
	log.debug(`starting session for account ${account.id} with ${persona.id}`)

	let session = await ctx.db.sessions.readOne({
		where: {
			bond: {
				account: account.id,
				persona: persona.id
			},
			timeEnded: null
		}
	})

	if(session){
		log.debug(`resuming session (${session.id})`)
	}

	if(!session){
		session = await ctx.db.sessions.createOne({
			data: {
				bond: {
					account: account.id,
					persona: persona.id
				}
			}
		})

		log.debug(`created new session (${session.id})`)
	}

	Object.assign(session, {
		llm: {
			primaryModel: llmModels[0]
		},
		messages: [],
		log: [],
	})

	return session
}

export async function getNextResponse({ ctx, session, userInput }){
	if(userInput){
		session.messages.push({
			role: 'user',
			...userInput
		})
	}

	return await new Promise((resolve, reject) => {
		let promise
		let emitResponse = response => {
			session.messages.push({
				role: 'avatar',
				...response
			})
			resolve(response)
		}

		if(session.messages.length === 0){
			promise = firstChatRound({ ctx, session, emitResponse })
		}else{
			promise = nextChatRound({ ctx, session, emitResponse })
		}

		promise.catch(error => {
			log.warn(`error while generating response:`, error)
			reject(error)
		})
	})
}

export function commitAvatarResponse({ ctx, session, response }){
	session.messages.push({
		role: 'avatar',
		text: response.text
	})

	return response
}