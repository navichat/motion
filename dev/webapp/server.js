import fs from 'fs'
import log from '@mwni/log'
import Koa from 'koa'
import Router from '@koa/router'
import serve from 'koa-static'
import range from 'koa-range'
import { createServer } from 'http'
import createChatServer from '@navichat/server'
import createSocketManager from '@mwni/wss'


const config = {
	cacheDir: './cache',
	resourcesDir: '../resources',
	computeEndpoint: 'wss://compute.navichat.ai',
	database: {
		host: 'navichat.ai',
		user: 'navichat',
		password: 'rWoryxAV85TPk141',
		database: 'navichat'
	}
}

log.info('*** navi web server ***')


const koa = new Koa()
const router = new Router()
const server = createServer(koa.callback())
const registerClient = createChatServer({ config })
const sockets = createSocketManager({
	server,
	authorize: ({ ip, headers }) => ({
		ip: headers['x-real-ip'] || ip,
		userAgent: headers['user-agent']
	})
})

sockets.on('accept', client => {
	registerClient(client)
})

router.get('/', async ctx => {
	let html = fs.readFileSync('./app.html', 'utf-8')
	let appJs = fs.readFileSync('./dist/app.js', 'utf-8')
	let appCss = fs.readFileSync('./dist/app.css', 'utf-8')

	ctx.body = html.replace('%appJs%', appJs).replace('%appCss%', appCss)
})

koa.use(router.routes(), router.allowedMethods())
koa.use(range)
koa.use(serve(config.resourcesDir))
koa.use(serve('./dist'))
koa.use(serve('./assets'))

server.listen(80)
log.info('listening on port 80')

console.errorOrg = console.error
console.error = text => /.*Error: (write|read) ECONN.*/g.test(text)
	? undefined
	: console.errorOrg(text)