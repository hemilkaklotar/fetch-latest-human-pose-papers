const fs = require('fs')
const csv = require('csv-parser')

const inputFile = 'papers.csv'
const outputFile = 'papers.md'

const createMarkdownLink = (text, link) => `[${text}](${link})`

fs.createReadStream(inputFile)
	.pipe(csv())
	.on('data', (row) => {
		const title = row['Title']
		const description = row['Description']
		const paperLink = row['Paper Link']
		const codeLink = row['Code Link']
		const stars = row['Stars']
		const datePublished = row['Date of Published']

		const markdown = `
## ${title}

${description}

- Paper: ${createMarkdownLink('Link', paperLink)}
- Code: ${createMarkdownLink('Link', codeLink)}
- Stars: ${stars}
- Published: ${datePublished}

`

		fs.appendFile(outputFile, markdown, (err) => {
			if (err) throw err
		})
	})
	.on('end', () => {
		console.log('Markdown file generated successfully!')
	})
