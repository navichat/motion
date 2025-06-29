export default {
	oncreate: node => {
		let wrapper = node.dom
		let canvas = node.dom.querySelector('canvas')
		let player = new node.attrs.app.playerClasses.Player(canvas)

		node.state.loadAvatar = async avatar => {
			let instance = await player.loadAvatar(avatar)

			player.cameraDriver.setView(avatar.previewCamera)
			instance.lookAtCameraAsIfHuman(player.camera)
			instance.setIdleAnimations({ 
				animations: avatar.previewAnimation,
				baseDuration: 10000,
				baseTime: Infinity
			})

			player.on('tick', instance.tick.bind(instance))
			node.state.instance = instance
			node.state.loaded = true
			m.redraw()
		}
		
		function resizeCanvas(){
			let w = wrapper.clientWidth
			let h = wrapper.clientHeight
			let r = window.devicePixelRatio

			canvas.width = w * r
			canvas.height = h * r
			canvas.style.width = `${w}px`
			canvas.style.height = `${h}px`

			player.handleCanvasResize()
		}

		resizeCanvas()
		window.addEventListener('resize', resizeCanvas)

		node.state.player = player
		node.state.resizeCanvas = resizeCanvas
	},
	onremove: node => {
		window.removeEventListener('resize', node.state.resizeCanvas)
		node.state.instance.dispose()
		node.state.player.dispose()
		console.log('disposed avatar preview')
	},
	view: ({ state, attrs }) => {
		if(state.avatar !== attrs.avatar){
			state.loaded = false
			state.avatar = attrs.avatar

			if(state.instance){
				state.player.off('tick')
				state.instance.dispose()
			}

			clearTimeout(state.loadTimeout)
			state.loadTimeout = setTimeout(
				() => state.loadAvatar(attrs.avatar),
				700
			)
		}

		return (
			<div class="avatar-preview">
				<canvas/>
				{
					!state.loaded && (
						<div class="loading">
							<img src="/icons/spinner.svg"/>
							<span>Loading {attrs.avatar.name}...</span>
						</div>
					)
				}
			</div>
		)
	}
}