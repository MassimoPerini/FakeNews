var request = require('request');

//Twitter API Keys
var consumerKey = "ncA4vA2bwaKuEaUVf6RJUfHON";
var consumerSecret = "5SMe0zFj9HzfIRG3gVkUEbH6MprggAGnPC970dvwgQSKAdtpfu";
var encodedConsumer = consumerKey+":"+consumerSecret;
var token = Buffer.from(encodedConsumer).toString('base64');
var bearer = null;



function getTwitterData(id, access_token, callback) {
	//Twitter show/:id API https://developer.twitter.com/en/docs/tweets/post-and-engage/api-reference/get-statuses-show-id
	var options = {
		url: "https://api.twitter.com/1.1/statuses/show.json?id="+id,
		method: "GET",
		headers: {
			"Authorization": "Bearer "+access_token,
		}
	}

	request(options, function (error, response, body){
		/*
		console.log("Twitter Token error: "+error);
		console.log("Twitter Token response: "+response);
		console.log("Twitter Token body: "+body);
		*/
		if (!error && response.statusCode == 200) {
			return callback(null, response);
		}
		return callback(error, "");
	});
}


module.exports = {
	getData: function(id,externalCallback) {
		module.exports.getTwitterToken(function(key){
			if(key)
				getTwitterData(id, key, function(error, response){
					externalCallback(error, response);
				});
		});
	},
	getTwitterToken: function (callback) {
	//Getting the Token for Application-Only authentication: https://developer.twitter.com/en/docs/basics/authentication/overview/application-only
	var options = {
		method: 'POST',
		uri: 'https://api.twitter.com/oauth2/token',
		headers: {
			'Authorization': 'Basic '+token,
			'content-type': 'application/x-www-form-urlencoded;charset=UTF-8'
		},
		body: 'grant_type=client_credentials'
	};

	request(options, function cb(error, response, body) {
		/*
		console.log("Twitter Token error: "+error);
		console.log("Twitter Token response: "+response);
		console.log("Twitter Token body: "+body);
		*/
		if (!error && response.statusCode == 200) {
			var info = JSON.parse(body)["access_token"];
			return callback(info);
		}
		return callback("");
	});
}
}

