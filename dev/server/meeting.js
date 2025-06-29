export async function provisionMeeting({ ctx, client, avatar }){
	let meeting = await ctx.db.meetings.readOne({
		where: {
			bond: {
				account: client.account.id,
				avatar
			},
			timeEnded: null
		}
	})

	if(!meeting){
		meeting = await ctx.db.meetings.createOne({
			data: {
				bond: {
					account: client.account.id,
					avatar
				}
			}
		})
	}

	Object.assign(meeting, {
		avatar,
		scene: 'classroom',
		messages: [],
		log: [],
	})

	return meeting
}