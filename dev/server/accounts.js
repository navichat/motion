import { hash, compare } from 'bcrypt'
import { generateRandomToken } from '@mwni/random'


const maxSignInAttemptsPer10m = 10
const maxRecoveryMailsPer10m = 5
const maxRecoveryAttemptsPer10m = 10


export async function readAccount({ ctx, id, sessionToken }){
	if(sessionToken){
		let session = await ctx.db.accountSessions.readOne({
			where: {
				token: sessionToken
			}
		})

		if(!session)
			return

		id = session.account.id
	}

	let account = await ctx.db.accounts.readOne({
		where: {
			id
		}
	})

	return account
}

export async function createGuestAccount({ ctx }){
	let account = await ctx.db.accounts.createOne({
		data: {
			guest: true,
			email: null,
			password: null,
			balance: 20
		}
	})

	return account
}

export async function createAccountSession({ ctx, account, userAgent, ip }){
	let token = generateRandomToken({
		segments: 1,
		charactersPerSegment: 16
	})

	let previousSession = await ctx.db.accountSessions.readOne({
		where: {
			account,
			userAgent,
			ip
		}
	})

	if(previousSession){
		//todo
		return previousSession
	}

	return await ctx.db.accountSessions.createOne({
		data: {
			account,
			token,
			userAgent,
			ip
		}
	})
}


/*export function initAccounts({ ctx }){
	async function requireAccount({ ctx }){
		let session = await getSessionFromCtx({ ctx })
	
		if(!session)
			throw {
				message: 'You are not signed in. Please refresh the page and sign in.',
				expose: true
			}
	
		return await db.accounts.readOne({
			where: {
				id: session.account.id
			}
		})
	}
	

	async function assertCredentials({ email, password }){
		let account = await db.accounts.readOne({
			where: {
				email
			}
		})
	
		if(!account){
			throw {
				message: 'No account with this email.',
				expose: true
			}
		}
	
		if(!await compare(password, account.password)){
			throw {
				message: 'Incorrect password.',
				expose: true
			}
		}
	}

	async function deleteSessionBy(where){
		await db.accountSessions.deleteOne({
			where
		})
	}

	async function createAccount({ email, password }){
		let account = await db.accounts.readOne({
			where: {
				email
			}
		})

		if(account){
			throw {
				message: 'This email is already reigstered.',
				expose: true
			}
		}
	
		let hashedPassword = await hash(password, 10)
	
		try{
			let account = await db.accounts.createOne({
				data: {
					email, 
					password: hashedPassword 
				}
			})

			await hooks.accountCreation({ account })
		}catch(error){
			throw {
				message: 'Accounts can not be created at this time.',
				expose: true
			}
		}
	}

	async function createRecovery({ email, userAgent, ip }){
		let account = await db.accounts.readOne({
			where: {
				email
			}
		})
	
		if(!account){
			throw {
				message: 'No account with this email.',
				expose: true
			}
		}
	
		try{
			let recovery = await db.accountRecoveries.createOne({
				data: {
					account, 
					code: generateCode({ digits: 6 }),
					userAgent,
					ip
				}
			})

			await hooks.recoveryCreation({ account, recovery })
		}catch(error){
			throw {
				message: 'Password reset emails cannot be sent out at the time.',
				expose: true
			}
		}
	}
		
	async function assertRecovery({ email, code }){
		let account = await db.accounts.readOne({
			where: {
				email
			}
		})

		if(!account){
			throw {
				message: 'No account with this email.',
				expose: true
			}
		}

		let recovery = await db.accountRecoveries.readOne({
			where: {
				account: {
					id: account.id
				},
				code
			}
		})

		if(!recovery){
			throw {
				message: 'Incorrect code.',
				expose: true
			}
		}

		return recovery
	}

	async function redeemRecovery({ accountkit, email, code, password }){
		let account = await db.accounts.readOne({
			where: {
				email
			}
		})

		if(!account){
			throw {
				message: 'No account with this email.',
				expose: true
			}
		}

		let recovery = await assertRecovery({
			accountkit,
			email,
			code
		})

		let hashedPassword = await hash(password, 10)

		await db.accountRecoveries.updateOne({
			data: {
				timeRedeemed: new Date()
			},
			where: {
				id: recovery.id
			}
		})

		await db.accounts.updateOne({
			data: {
				password: hashedPassword
			},
			where: {
				id: account.id
			}
		})

		await hooks.passwordChange({ account })
	}

	Object.assign(ctx.methods, {
		readAccount,
		createGuestAccount,
		createAccountSession
	})
}*/