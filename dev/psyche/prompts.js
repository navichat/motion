import fs from 'fs'
import path from 'path'
import util from 'util'
import { fileURLToPath } from 'url'
import { parse } from 'yaml'


const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

function compileTemplates(dict){
	for(let [key, value] of Object.entries(dict)){
		if(typeof value === 'object'){
			dict[key] = compileTemplates(value)
		}else{
			dict[key] = new Template(value)
		}
	}

	return dict
}

export class Template extends String{
	format(fields){
		let str = this.toString()

		for(let [key, value] of Object.entries(fields)){
			str = str.replaceAll(`{${key}}`, value)
		}

		return new Template(str)
	}

	get [Symbol.toStringTag](){
		return this.toString()
	}

	[util.inspect.custom]() {
		return `\x1b[32m'${this.toString()}\x1b[0m'`
	}
}

export default compileTemplates(
	parse(
		fs.readFileSync(
			path.join(__dirname, 'prompts.yml'),
			'utf-8'
		)
	)
)