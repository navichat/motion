import anime from 'animejs'

export default {
	oncreate: async node => {
		let popScale = 0.1 * (node.attrs.popScale || 1)

		await Promise.resolve()
		
		node.dom.addEventListener('mouseenter', () => {
			navi.sounds.drip.play()
			anime({
				targets: [node.dom],
				scaleX: 1 + popScale,
				scaleY: 1 + popScale,
				duration: 300
			})
		})

		node.dom.addEventListener('mouseleave', () => {
			anime({
				targets: [node.dom],
				scaleX: 1,
				scaleY: 1,
				duration: 300
			})
		})

		node.dom.addEventListener('mousedown', () => {
			navi.sounds[node.attrs.clickSound || 'click'].play()
			anime({
				targets: [node.dom],
				scaleX: 1 - popScale,
				scaleY: 1 - popScale,
				duration: 40,
				easing: 'linear'
			})
		})

		node.dom.addEventListener('mouseup', () => {
			anime({
				targets: [node.dom],
				scaleX: 1,
				scaleY: 1,
				duration: 300
			})
		})
	},
	view: node => node.children
}