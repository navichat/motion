import NoAudioOverlay from './NoAudioOverlay.jsx'
import LoadingScreen from './LoadingScreen.jsx'
import Home from './Home.jsx'

export default ({ attrs }) => ({
	view: node => (
		<>
			{ attrs.app.state.audioBlocked ? <NoAudioOverlay/> : null }
			{ attrs.app.state.loading ? <LoadingScreen/> : null }
			{ attrs.app.state.atHome ? <Home app={attrs.app}/> : null }
		</>
	)
})