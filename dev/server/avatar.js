import { getNextPsycheResponse } from '@navichat/psyche'

export async function generateNextAvatarAction({ ctx, meeting }){
	let { text, tone } = await getNextPsycheResponse({ ctx, meeting })

	meeting.messages.push({
		role: 'avatar',
		text
	})

	return {
		type: 'speech',
		text
	}
}