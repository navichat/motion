import { get } from '@mwni/fetch'

export default async ({ app }) => {
	let sceneSpec = await get({ url: '/scenes/village_street.json' })
	let cameraDriver = new app.playerClasses.FreeCameraDriver()

	cameraDriver.setView(sceneSpec.defaultCamera)
	app.player.cameraDrivers.push(cameraDriver)

	await app.player.loadScene(sceneSpec)

	return {
		dispose(){
			app.player.cameraDrivers.splice(app.player.cameraDrivers.indexOf(cameraDriver), 1)
			app.player.clear()
		}
	}
}