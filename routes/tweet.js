var url = require("url");
var isUrl = require("is-url");
var express = require('express');
var twitter = require('../utils/twitter');
var embedder = require('../utils/embedder');
var router = express.Router();

/* GET tweet page. */
router.post('/tweet', function(req, res){
	if (isUrl(req.body.tweet)) {
		var ids=url.parse(req.body.tweet).pathname.split('/');
		twitter.getData(ids[ids.length-1], function callback(error, response){
			if (!error && response.statusCode == 200){
				console.log("Got inside twitter data.");
				console.log(response);
				embedder.getHTML(ids[ids.length-1], function(htmlResponse){
					console.log("Got inside embedder");
					console.log(htmlResponse);
					return res.send(htmlResponse);	
				});
			} 
			if(error != null){
				console.log("Error getting data: "+error);
				return res.send("Not a tweet: "+error);
			}
		});
	}else{
		console.log("Not a url");
		res.send("Not a url");
	}
});

module.exports = router;