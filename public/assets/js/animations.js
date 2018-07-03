var assignOnFunctions = function() {
	let prefix = ".tweet"
	$(".tweet0").on('click', { prefix: ".tweet", care: "0", dontcare: ["1", "2"] }, demotweet)
	$(".tweet1").on('click', { prefix: ".tweet", care: "1", dontcare: ["0", "2"] }, demotweet)
	$(".tweet2").on('click', { prefix: ".tweet", care: "2", dontcare: ["0", "1"] }, demotweet)
}
var demotweet = function(e) {
	let care = e.data.prefix.concat(e.data.care)
	$(care).off();
	$(care).toggleClass('positionZero');
	for (let dc of e.data.dontcare) {
		dc = e.data.prefix.concat(dc)
		$(dc).fadeOut(500)
	}
	$(".resultLoaded").hide();
	$(".response").fadeIn(500)
	setTimeout(function() {
		$(".part").hide();
		$(".resultLoadedTweet" + e.data.care).show();
		$("#goToTweet").attr("href", "https://twitter.com/Twitter/status/".concat($(".resultLoadedTweet" + e.data.care).attr('id')));
		$(".resultLoadedButtons").show();


		$(care).on('click', quitAllDemo);
	}, 3000);
}
var quitAllDemo = function() {
	let twClasses = [".tweet0", ".tweet1", ".tweet2"]
	for (twClass of twClasses) {
		$(twClass).off()
		$(twClass).fadeIn(500)
		$(twClass).removeClass('positionZero')
	}
	$(".part").toggle();
	$(".response").hide()
	$(".resultLoaded").fadeOut(1000)
	assignOnFunctions()
}

$(document).ready(function() {
	assignOnFunctions()
	$("#quitDemo").on("click", quitAllDemo);
});