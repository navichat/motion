import log from '@mwni/log'
import { Template } from './prompts.js'

export const models = [
	{
		id: 'mixtral-8x7b-instruct-v0.1-6bit'
	}
]

export async function llmComplete({ ctx, thread = [], system, prompt, preface, fill, mode, max_tokens = 250,  }){
	let stoppingRegex

	if(mode === 'sentence'){
		stoppingRegex = '(\\.|\\!|\\?)\\s*$'
	}else if(mode === 'phrase'){
		stoppingRegex = '(\\,|\\.|\\!|\\?)\\s*$'
	}else if(mode === 'quote'){
		stoppingRegex = '(\\")\\s*$'
	}

	let threadCopy = composeThreadForQuery({ thread, prompt, fill })
	let { text } = await llmQuery({
		ctx,
		command: 'llm_chat',
		stopping_regex: stoppingRegex,
		messages: threadCopy.forQuery({ system, preface })
	})

	threadCopy.appendResult({ preface, text })

	return threadCopy
}

export async function llmLogits({ ctx, thread = [], system, prompt, preface, fill, probe }){
	let { logits } = await llmQuery({
		ctx,
		command: 'llm_chat_logits',
		messages: composeThreadForQuery({ thread, prompt, fill })
			.forQuery({ system, preface }),
		logits_for: probe
	})

	return logits
}

async function llmQuery({ ctx, command, messages, ...query }){
	let model = ctx.session.llm.primaryModel
	let xid = Math.random()
		.toString(16)
		.slice(2)
	
	log.time.debug(`llm.query.${xid}`, `querying ${model.id} (${command})`)

	let result = await ctx.compute.request({
		command,
		model: model.id,
		messages,
		...query
	})

	log.time.debug(`llm.query.${xid}`, `querying ${model.id} (${command}) took %`)

	ctx.log.push({
		type: 'llm',
		command,
		time: Date.now(),
		model: model.id,
		messages,
		...query,
		result
	})

	return result
}

function composeThreadForQuery({ thread, prompt, fill }){
	if(prompt)
		thread = [...thread, { user: prompt }]

	thread = new LLMThread(...thread)

	if(fill)
		thread = thread.fill(fill)

	return thread
}


export class LLMThread extends Array{
	constructor(...messages){
		if(typeof messages[0] === 'number')
			return new Array(messages[0])

		super(...messages.map(Message.from))
	}

	get last(){
		return this[this.length - 1]
	}

	fill(data){
		return new LLMThread(
			...this.map(
				message => message.format(data)
			)
		)
	}

	forQuery({ system, preface }){
		let messages = this.map(message => message.toRole())

		if(system)
			messages.unshift({ role: 'system', text: system })

		if(preface)
			messages.push({ role: 'assistant', text: preface })

		return messages
	}

	appendResult({ preface, text }){
		this.push(new Message({ 
			assistant: { 
				preface, 
				text
			}
		}))
	}
}

class Message extends Template{
	static from(value){
		if(value instanceof Message)
			return value

		return new Message(value)
	}

	constructor({ system, user, assistant }){
		if(system){
			super(system)
			this.role = 'system'
		}else if(user){
			super(user)
			this.role = 'user'
		}else if(assistant){
			if(typeof assistant === 'string' || !assistant.text)
				super(assistant)
			else{
				super((assistant.preface || '') + assistant.text)
				this.preface = assistant.preface
				this.withoutPreface = assistant.text.trim()
			}
			this.role = 'assistant'
		}else{
			throw new Error(`Message must be constructed with either "system", "user" or "assistant" string`)
		}
	}

	format(data){
		return new Message({
			[this.role]: this.preface
				? { preface: this.preface, result: super.format(data).trim() }
				: super.format(data).trim()
		})
	}

	toRole(){
		return {
			role: this.role,
			text: this.toString()
		}
	}
}