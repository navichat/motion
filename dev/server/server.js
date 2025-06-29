import log from '@mwni/log'
import createCloudkitClient from '@cloudkit/client'
import { open as openDb } from '@structdb/mysql'
import { loadResources, loadAnimations } from './engine-stub.js'
import { readAccount, createGuestAccount, createAccountSession } from './accounts.js'
import { provisionMeeting } from './meeting.js'
import schema from './schema.json' assert { type: 'json' }
import { generateNextAvatarAction } from './avatar.js'


export default ({ config }) => {
	log.info('initializing')

	let ctx = {
		paths: {
			cache: config.cacheDir,
			resources: config.resourcesDir
		},
		compute: createCloudkitClient({ 
			url: config.computeEndpoint
		}),
		db: openDb({
			credentials: config.database,
			poolSize: 10,
			schema
		}),
		clients: []
	}

	log.info(`using compute endpoint ${config.computeEndpoint}`)
	log.info(`using db at ${config.database.host}`)

	loadAnimations({ ctx })
	loadResources({ ctx })

	ctx.compute.on('connect', () => {
		log.info('connected to compute')
	})

	ctx.compute.on('error', error => {
		log.warn(`connection to compute failed: ${error.message}`)
	})

	return client => {
		ctx.clients.push(client)

		client.on('hello', async ({ sessionToken }) => {
			client.account = await readAccount({ ctx, sessionToken })
	
			if(!client.account){
				client.account = await createGuestAccount({ ctx })
	
				let session = await createAccountSession({
					ctx,
					account: client.account,
					ip: client.ip,
					userAgent: client.userAgent
				})

				sessionToken = session.token
			}else{
				
			}

			client.send({
				event: 'welcome',
				sessionToken,
				account: {
					guest: client.account.guest,
					balance: client.account.balance
				},
			})

			log.info(`client hello with account ${client.account.id}`)
		})

		client.on('meeting', async ({ id, avatar }) => {
			log.info(`client ${client.account.id} enters meeting with ${avatar}`)

			client.meeting = await provisionMeeting({ ctx, client, avatar })

			client.send({
				event: 'meeting',
				id,
				avatar: ctx.resources.avatars[client.meeting.avatar],
				scene: ctx.resources.scenes[client.meeting.scene],
				firstAvatarAction: await generateNextAvatarAction({ 
					ctx, 
					meeting: client.meeting 
				})
			})
		})

		client.on('disconnect', ({ code }) => {
			ctx.clients.splice(ctx.clients.indexOf(client), 1)
			log.info(`client disconnect with account ${client.account?.id} (code ${code})`)
		})
	}
}