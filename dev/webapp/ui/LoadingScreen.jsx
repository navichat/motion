import anime from 'animejs'

export default {
	oncreate: ({ dom }) => {
		let backdrop = dom.querySelector('.backdrop')
		let brandParts = dom.querySelector('.brand').children
		let status = dom.querySelector('.status')

		let startTime = performance.now()
		let tickFloat = () => {
			let t = performance.now() - startTime

			for(let i=0; i<brandParts.length; i++){
				let y = Math.cos(Math.PI * 0.6 + t / 2000 - i / 4) * 17
				brandParts[i].children[0].style.transform = `translateY(${y}px)`
			}

			requestAnimationFrame(tickFloat)
		}

		anime({
			targets: [backdrop],
			opacity: [0, 1],
			duration: 5000,
			easing: 'easeOutSine'
		})

		anime({
			targets: [status],
			opacity: [0, 1],
			duration: 1000,
			delay: 1000,
			easing: 'easeOutSine'
		})

		for(let i=0; i<brandParts.length; i++){
			anime({
				targets: [brandParts[i]],
				translateY: [50, 0],
				opacity: [0, 1],
				duration: 3000,
				delay: i * 80,
				easing: 'easeOutExpo'
			})
		}

		tickFloat()
	},
	view: node => (
		<div class="loading-screen">
			<div class="backdrop"/>
			<div class="content">
				<div class="brand">
					<div><img src="icons/brand_part_logo.svg"/></div>
					<div><img src="icons/brand_part_n.svg"/></div>
					<div><img src="icons/brand_part_a.svg"/></div>
					<div><img src="icons/brand_part_v.svg"/></div>
					<div><img src="icons/brand_part_i.svg"/></div>
				</div>
				<div class="status">
					<span>Loading Assets...</span>
				</div>
			</div>
		</div>
	)
}