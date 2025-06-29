import fs from 'fs'
import { fork } from 'child_process'
import esbuild from 'esbuild'
import postcss from 'postcss'
import postcssImport from 'postcss-import'
import postcssUnnest from 'postcss-nested'
import chokidar from 'chokidar'
import log from '@mwni/log'

let server
let postcssCompiler = postcss([
	postcssImport,
	postcssUnnest
])

function spawnServer(){
	if(server){
		server.kill()
		log.info('restarting server')
	}else{
		log.info('starting server')
	}

	server = fork('server.js')
}

function compileCss(){
	log.info('compiling css...')

	postcssCompiler.process(fs.readFileSync('./app.scss', 'utf-8'), { map: false, from: '.' })
		.then(({ css, messages }) => {
			fs.writeFileSync('./dist/app.css', css)

			for(let message of messages){
				log.warn(message)
			}

			log.info('css bundle complete')
		})
		.catch(error => {
			log.error('css error:', error)
		})
}

compileCss()

let ctx = await esbuild.context({
	entryPoints: ['./app.js', './player.loader.js'],
	jsxFactory: 'm',
	jsxFragment: '\'[\'',
	outdir: './dist',
	treeShaking: true,
	bundle: true,
	plugins: [
		{
			name: 'spawn-server-on-build',
			setup(build){
				build.onStart(() => {
					log.info('building app...')
				})
				build.onEnd(() => {
					log.info('app build complete')
					spawnServer()
				})
			}
		}
	]
})

chokidar.watch(['./server.js', '../server', '../psyche', '../../engine/player', '../resources']).on('change', path => {
	log.info(`file changed: ${path}`)
	spawnServer()
})

chokidar.watch(['app.scss']).on('change', path => {
	log.info(`file changed: ${path}`)
	compileCss()
})

await ctx.watch()