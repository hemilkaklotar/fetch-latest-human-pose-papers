const axios = require('axios')
const cheerio = require('cheerio')
const fs = require('fs')
const { Parser } = require('json2csv')

const baseUrl = 'https://paperswithcode.com'
const url = 'https://paperswithcode.com/task/human-motion-prediction/latest'
const data = []

async function scrapeData() {
	try {
		let page = 1
		let year = 2023

		while (year >= 2018) {
			const response = await axios.get(`${url}?page=${page}`)

			const $ = cheerio.load(response.data)

			$('.paper-card').each((index, element) => {
				if (index === 0) {
					fs.writeFileSync('indx.html', $('.entity').toString())
				}
				const title = JSON.stringify(
					$(element).find('.item-content h1').text().trim(),
				)
				const description = JSON.stringify(
					$(element).find('.item-content .item-strip-abstract').text().trim(),
				)
				const entity = $(element).find('.entity')
				const paperLink =
					baseUrl + (entity.find('.badge-light').attr('href') || '')
				const codeLink =
					baseUrl + (entity.find('.badge-light').attr('href') || '')

				const stars = $(element)
					.find('.entity-stars')
					.find('.badge-secondary')
					.text()

				const dateOfPublished = $(element)
					.find('.stars-accumulated.text-center')
					.text()
					.trim()

				data.push({
					Title: title,
					Description: description,
					'Paper Link': paperLink,
					'Code Link': codeLink,
					Stars: stars,
					'Date of Published': dateOfPublished,
				})
			})

			const nextPageLink = $('.pagination li:last-child a').attr('href')

			if (!nextPageLink || nextPageLink.includes('page=1')) {
				year--
			}

			page++
		}

		saveToCSV(data)
	} catch (error) {
		console.error('Error:', error)
	}
}

function saveToCSV(data) {
	const json2csvParser = new Parser()
	const csv = json2csvParser.parse(data)

	fs.writeFileSync('papers.csv', csv, 'utf-8')

	console.log('Data saved to papers.csv')
}

scrapeData()
