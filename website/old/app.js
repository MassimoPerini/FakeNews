/** Express app configuration file. Also loads all dependencies and connects to DB.
 * @module app
 */

/* Express */
var express = require('express');
var path = require('path');
var favicon = require('serve-favicon');
var cookieParser = require('cookie-parser');
var bodyParser = require('body-parser');
var HttpErr = require('http-errors');
/* Passport auth */
var session = require('express-session');
/* Logging */
var logger = require('morgan');
var debug = require('debug')('app');
/* API Documentation */
var swaggerJSDoc = require('swagger-jsdoc');

/* Basic express app setup */
var app = express();
// Port setup
app.set('port', (process.env.PORT || 8000));
// View setup
app.set('views', path.join(__dirname, 'views'));
app.set('view engine', 'ejs');
// uncomment after placing your favicon in /public
//app.use(favicon(path.join(__dirname, 'public', 'favicon.ico')));
app.use(logger('dev'));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(cookieParser());
// Static directories
app.use('/public',express.static(path.join(__dirname, 'public')));
app.use('/docs',express.static(path.join(__dirname, 'docs')));


/* API Documentation config using swagger*/
var swaggerSpec = swaggerJSDoc({
  swaggerDefinition: {
    info: {
      title: 'Fake News API',
      version: '0.0.1',
      description: 'Fake News website API documentation, part of ...'
    },
    host: 'http://www.url_website.xyz',
    basePath: '/api/'
  },
  apis: ['./routes/*.js']
});

/* Routers */
var index = require('./routes/index');
app.use('/', index);

// Serve swagger json
app.get('/docs/api/swagger.json', function(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.send(swaggerSpec);
});

/* Error handling */
// Catch 404 errors
app.use(function(req, res, next) {
  next(new HttpErr.NotFound());
});
// Error message (development and production)
app.use(function(err, req, res, next) {
  // set locals, only providing error in development
  res.locals.message = err.message;
  res.locals.error = req.app.get('env') === 'development' ? err : {};
  res.status(err.status || 500);
  res.render('error', {user: req.user});
});

/* Serve app */
app.listen(app.get('port'), function() {
    debug('Node app is running on port', app.get('port'));
});

module.exports = app;
