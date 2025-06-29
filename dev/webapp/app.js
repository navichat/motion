import m from 'mithril'
import anime from 'animejs'
import { Howl } from 'howler'
import createSocket from '@mwni/socket'
import createHome from './home.js'
import createMeeting from './meeting.js'
import MainComponent from './ui/Main.jsx'


class App{
	constructor(){
		this.dom = {}
		this.state = { 
			audioBlocked: true,
			loading: true,
			bootstrapped: false
		}
		this.initClient()
		this.initDom()
		this.initUI()
		this.initSounds()
	}

	initClient(){
		this.socket = createSocket({
			url: `ws${location.protocol.slice(4)}${location.hostname}`
		})

		this.socket.on('connect', () => {
			console.log('connection established')

			this.socket.send({ 
				command: 'hello',
				sessionToken: localStorage.getItem('sessionToken')
			})
		})

		this.socket.on('disconnect', ({ code, reason }) => {
			console.warn('connection lost', code, reason)
			
			if(this.meeting){
				this.leaveMeeting()
			}
		})

		this.socket.on('error', error => {
			console.error('connection error', error)
		})

		this.socket.on('welcome', ({ sessionToken }) => {
			console.log('welcome from server 😊')

			localStorage.setItem('sessionToken', sessionToken)

			if(!this.state.bootstrapped){
				this.initBootSequence(
					//this.loadHome()
					this.enterMeeting({ avatarId: 'ichika' })
				)
				this.state.bootstrapped = true
			}
		})
	}

	initDom(){
		let canvas = document.createElement('canvas')
		document.body.appendChild(canvas)

		let ui = document.createElement('div')
		ui.className = 'ui'
		document.body.appendChild(ui)

		this.dom = {
			canvas,
			ui
		}
	}

	initUI(){
		m.mount(this.dom.ui, {
			view: () => m(MainComponent, {
				app: this
			})
		})
	}

	initSounds(){
		let handleFirstClick = () => {
			this.activateSounds()
			window.removeEventListener('click', handleFirstClick)
		}

		this.sounds = {
			theme: new Howl({ src: ['/sounds/theme.ogg'], html5: true, loop: true }),
			drip: new Howl({ src: ['/sounds/drip.ogg'] }),
			click: new Howl({ src: ['/sounds/click.ogg'] }),
			activate: new Howl({ src: ['/sounds/activate.ogg'] }),
			submit: new Howl({ src: ['/sounds/submit.ogg'] }),
		}

		window.addEventListener('click', handleFirstClick)
	}

	activateSounds(){
		//this.sounds.theme.play()
		this.state.audioBlocked = false
		m.redraw()
	}

	async initBootSequence(promise){
		await promise

		await anime({
			targets: [this.dom.ui.querySelector('.loading-screen')],
			opacity: 0,
			duration: 1000,
			easing: 'easeInCubic'
		}).finished

		this.state.loading = false
		m.redraw()

		await anime({
			targets: [this.dom.canvas],
			opacity: [0, 1],
			duration: 3000,
			easing: 'easeOutCubic'
		}).finished
	}

	async loadHome(){
		this.home = await createHome({ app: this })
	}

	async enterMeeting({ avatarId }){
		this.player.freeze()

		if(this.home){
			this.home.dispose()
			this.home = undefined
		}

		this.meeting = await createMeeting({
			app: this,
			config: await this.socket.request({
				command: 'meeting',
				expectEvent: 'meeting',
				avatar: avatarId
			})
		})

		this.player.unfreeze()
	}

	async leaveMeeting(){
		this.player.freeze()

		this.meeting.dispose()
		this.meeting = undefined

		await this.loadHome()

		this.player.unfreeze()
	}

	registerPlayer(classes){
		this.playerClasses = classes
		this.player = new classes.Player(this.dom.canvas)
		this.autoResizeCanvas()
	}

	autoResizeCanvas(){
		window.addEventListener('resize', () => this.player.updateSize())
	}
}

window.m = m
window.navi = new App()