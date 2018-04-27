/**
  Index.js file loads all the files containing
  express routers inside the routes directory
  and uses them in the main application.
**/

var express = require('express');
var router = express.Router();

var path = require("path");
var fs = require('fs');

router.get('/', function(req, res, next) {
  res.render('index', { title: 'Demo Website', user:req.user });
});

fs.readdirSync(__dirname)
.filter(function(file) {
  return (file.indexOf(".") !== 0) && (file !== "index.js");
})
.forEach(function(file) {
  var subrouter = require(path.join(__dirname, file));
  router.use(subrouter);
});

/* --- WIP Pages --- */
// These are pages that are browseable (in the navbar or with link) but are yet to be implemented
// Given that they were not required, we did not implement these due to the time factor and
router.get(['/infos', '/news', '/faq'], function(req, res, next){
  res.render('blank', {title: 'Gallery', user:req.user});
});


module.exports = router;
