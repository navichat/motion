import anime from 'animejs'
import AnimatedButton from './AnimatedButton.jsx'
import AvatarPreview from './AvatarPreview.jsx'

export default {
	onbeforeremove: async node => {
		let backdrop = node.dom.querySelector('.backdrop')
		let content = node.dom.querySelector('.content')

		node.dom.querySelector('.avatar-preview').style.visibility = 'hidden'

		await Promise.all([
			anime({
				targets: [node.dom],
				backgroundColor: 'rgba(255, 255, 255, 0)',
				duration: 500,
				easing: 'easeOutExpo'
			}).finished,
			anime({
				targets: [content, backdrop],
				translateX: -content.clientWidth * 2,
				duration: 700,
				easing: 'easeOutExpo'
			}).finished
		])
		
	},
	view: node => (
		<div class={`home ${node.state.selectedAvatar && 'in-preview'}`}>
			<div class="backdrop"/>
			<div class="content">
				<div class="header">
					<img class="brand" src="/icons/brand.svg"/>
				</div>
				<div class="section">
					<h2>Pick your conversation partner</h2>
					<div class="avatars">
						{
							node.attrs.app.avatars.new.map(
								avatar => <AvatarThumb 
									avatar={avatar}
									onselect={() => node.state.selectedAvatar = avatar}
									selected={avatar === node.state.selectedAvatar}
									disabled={node.state.selectedAvatar && avatar !== node.state.selectedAvatar}
								/>
							)
						}
						<>
						{
							node.state.selectedAvatar
								? <AvatarDetails 
									key={node.state.selectedAvatar.id} 
									avatar={node.state.selectedAvatar}
									onclose={() => node.state.selectedAvatar = null}
									app={node.attrs.app}
								/>
								: null
						}
						</>
					</div>
				</div>
			</div>
			{
				node.state.selectedAvatar && !node.state.removing ?
					<AvatarPreview
						app={node.attrs.app}
						avatar={node.state.selectedAvatar}
					/>
					: null
			}
			
		</div>
	)
}

const AvatarThumb = {
	view: ({ attrs }) => (
		<AnimatedButton>
			<div class="thumb" onclick={attrs.onselect} selected={attrs.selected ? 'selected' : null}>
				<div
					class="thumbnail"
					style={`background-image: url(${attrs.avatar.thumbnail}); border-color: ${attrs.avatar.color[attrs.selected ? 1: 0]};`}
				/>
				<span style={`color: ${attrs.avatar.color[1]}`}>{attrs.avatar.name}</span>
			</div>
		</AnimatedButton>
	) 
}

const AvatarDetails = {
	oncreate: node => {
		anime({
			targets: [node.dom.querySelector('h1')],
			translateX: [-50, 0],
			opacity: [0, 1],
			duration: 900,
			easing: 'easeOutExpo'
		})
		anime({
			targets: [node.dom.querySelector('h2')],
			translateX: [-50, 0],
			opacity: [0, 1],
			duration: 900,
			delay: 100,
			easing: 'easeOutExpo'
		})
		anime({
			targets: [node.dom.querySelector('span')],
			translateY: [-10, 0],
			opacity: [0, 1],
			duration: 400,
			delay: 300,
			easing: 'easeOutExpo'
		})
		anime({
			targets: [node.dom.querySelector('button.chat')],
			translateY: [-10, 0],
			opacity: [0, 1],
			duration: 400,
			delay: 500,
			easing: 'easeOutExpo'
		})
		anime({
			targets: [node.dom.querySelector('button .heart')],
			scale: [1, 1.5, 1.2, 1.05],
			duration: 300,
			delay: 2000,
			loop: true,
			easing: 'linear'
		})
		anime({
			targets: [node.dom],
			height: [0, undefined],
			duration: 600,
			easing: 'easeOutExpo'
		})
	},
	view: ({ attrs }) => (
		<div class="details" style={`border-color: ${attrs.avatar.color[0]};`}>
			<button class="close" onclick={attrs.onclose}>✖</button>
			<h1>{attrs.avatar.name}</h1>
			<h2>{attrs.avatar.slogan}</h2>
			<span>{attrs.avatar.description}</span>
			<AnimatedButton popScale={0.5} clickSound="activate">
				<button 
					class="chat" 
					onclick={() => attrs.app.startSession({ avatar: attrs.avatar })} 
					style={`background-color: ${attrs.avatar.color[1]}; box-shadow: inset 2px 6px 20px ${attrs.avatar.color[0]};`}
				>
					<span>Start chatting</span>
					<span class="heart">♥</span>
				</button>
			</AnimatedButton>
		</div>
	)
}