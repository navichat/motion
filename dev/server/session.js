import { createEmitter } from '@mwni/events'
import { generateIdleAnimations } from './animation/stock.js'
import { generateSpeech } from './pipelines/speech.js'

export function createSession({ ctx, connection }){
	let session = {
		...createEmitter(),
		connection
	}

	async function init(){
		Object.assign(session, {
			avatar: ctx.resources.avatars.ichika,
			scene: ctx.resources.scene.classroom
		})

		connection.send({
			event: 'init',
			avatar: session.avatar,
			scene: session.scene
		})
	
		connection.send({
			event: 'idle',
			animations: generateIdleAnimations({
				ctx,
				longIdle: false 
			}),
			baseDuration: 3000
		})

		setTimeout(sayHiToDev, 5000)
	}

	async function sayHiToDev(){
		let speech = await generateSpeech({
			ctx,
			text: `Pop pop pop. --- Tick --- tock --- tick --- tock -- pop -- pop -- pop`,
			avatar: session.avatar
		})

		connection.send({
			event: 'speech',
			audio: speech.audio.toString('base64'),
			animation: {
				face: speech.face,
				body: speech.body,
			}
		})
	}

	init()

	return session
}