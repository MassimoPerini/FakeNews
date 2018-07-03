var express = require('express');
var jsonfile = require('jsonfile')
var pick = require('random-pick')


var router = express.Router();
var demoTweets = jsonfile.readFileSync('demoTweets.json', 'utf8')

var embedder = require('../utils/embedder');
//Example
//getHTML(idTweet, callback)

/* GET home page. */
router.get('/', function(req, res, next) {
	res.render('index', {
		title: 'Dr.Watson',
		tag_title_description: 'The Fake News Inspector',
		description: 'An AI-based platform to check fake news on twitter via Watson IBM',
		demoTweets: pick(demoTweets, 3),
		sentimentParser: {
			anger: "Anger 😡",
			fear: "Fear 😱",
			joy: "Joy 😁",
			sadness: "Sadness 😞",
			analytical: "Analytical 🤔",
			confident: "Confident 😎",
			tentative: "Tentative 😉",
			disgust: "Disgust 🤢"
		}
	});
});

module.exports = router;