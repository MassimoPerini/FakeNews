var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
	res.render('index', {
		title: 'Dr.Watson',
		tag_title_description: 'The fake news huter',
		description: 'An AI-based platform to check fake news on twitter via Watson IBM'
	});
});

module.exports = router;