var Twitter = require('twitter');
var tokenTweet = require('./twitter');

module.exports = {
	getHTML: function(id,externalCallback) {
		tokenTweet.getTwitterToken(function (accessToken){
			var client = new Twitter({
				consumer_key: 'ncA4vA2bwaKuEaUVf6RJUfHON',
				consumer_secret: '5SMe0zFj9HzfIRG3gVkUEbH6MprggAGnPC970dvwgQSKAdtpfu',
				bearer_token: accessToken
			});	

			client.get('statuses/oembed',{url: 'https://twitter.com/Interior/status/'+id}, function(error, response, raw) {
				if(error) throw error;
			//console.log(response);
			externalCallback(response["html"]);
		});	
		})

	}	
}