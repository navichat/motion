
export default async ({ app, config }) => {
	let cameraDriver = new app.playerClasses.AvatarFacingCameraDriver()
	let scene = await app.player.loadScene(config.scene)
	let avatar = await app.player.loadAvatar(config.avatar)
	let tick = delta => {
		avatar.tick(delta)
	}
	
	avatar.setRootTransform(config.scene.avatarRoot)
	avatar.lookAtCameraAsIfHuman(app.player.camera)
	
	cameraDriver.configure({ avatar, ...config.avatar.facingCamera })
	app.player.cameraDrivers.push(cameraDriver)

	app.player.on('tick', tick)

	return {
		dispose(){
			app.player.off('tick', tick)
			app.player.cameraDrivers.splice(app.player.cameraDrivers.indexOf(cameraDriver), 1)
			app.player.clear()
		}
	}
}