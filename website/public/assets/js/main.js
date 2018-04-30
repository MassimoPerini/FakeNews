/*
	Hyperspace by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/

(function($) {

	skel.breakpoints({
		xlarge: '(max-width: 1680px)',
		large: '(max-width: 1280px)',
		medium: '(max-width: 980px)',
		small: '(max-width: 736px)',
		xsmall: '(max-width: 480px)'
	});

	$(function() {

		var $window = $(window),
			$body = $('body'),
			$sidebar = $('#sidebar');

		// Hack: Enable IE flexbox workarounds.
		if (skel.vars.IEVersion < 12)
			$body.addClass('is-ie');

		// Disable animations/transitions until the page has loaded.
		if (skel.canUse('transition'))
			$body.addClass('is-loading');

		$window.on('load', function() {
			window.setTimeout(function() {
				$body.removeClass('is-loading');
			}, 100);
		});

		// Forms.

		// Fix: Placeholder polyfill.
		$('form').placeholder();

		// Hack: Activate non-input submits.
		$('form').on('click', '.submit', function(event) {

			// Stop propagation, default.
			event.stopPropagation();
			event.preventDefault();

			// Submit form.
			$(this).parents('form').submit();

		});

		// Prioritize "important" elements on medium.
		skel.on('+medium -medium', function() {
			$.prioritize(
				'.important\\28 medium\\29',
				skel.breakpoint('medium').active
			);
		});

		// Sidebar.
		if ($sidebar.length > 0) {

			var $sidebar_a = $sidebar.find('a');

			$sidebar_a
				.addClass('scrolly')
				.on('click', function() {

					var $this = $(this);

					// External link? Bail.
					if ($this.attr('href').charAt(0) != '#')
						return;

					// Deactivate all links.
					$sidebar_a.removeClass('active');

					// Activate link *and* lock it (so Scrollex doesn't try to activate other links as we're scrolling to this one's section).
					$this
						.addClass('active')
						.addClass('active-locked');

				})
				.each(function() {

					var $this = $(this),
						id = $this.attr('href'),
						$section = $(id);

					// No section for this link? Bail.
					if ($section.length < 1)
						return;

					// Scrollex.
					$section.scrollex({
						mode: 'middle',
						top: '-20vh',
						bottom: '-20vh',
						initialize: function() {

							// Deactivate section.
							if (skel.canUse('transition'))
								$section.addClass('inactive');

						},
						enter: function() {

							// Activate section.
							$section.removeClass('inactive');

							// No locked links? Deactivate all links and activate this section's one.
							if ($sidebar_a.filter('.active-locked').length == 0) {

								$sidebar_a.removeClass('active');
								$this.addClass('active');

							}

							// Otherwise, if this section's link is the one that's locked, unlock it.
							else if ($this.hasClass('active-locked'))
								$this.removeClass('active-locked');

						}
					});

				});

		}

		// Scrolly.
		$('.scrolly').scrolly({
			speed: 1000,
			offset: function() {

				// If <=large, >small, and sidebar is present, use its height as the offset.
				if (skel.breakpoint('large').active &&
					!skel.breakpoint('small').active &&
					$sidebar.length > 0)
					return $sidebar.height();

				return 0;

			}
		});

		// Spotlights.
		$('.spotlights > section')
			.scrollex({
				mode: 'middle',
				top: '-10vh',
				bottom: '-10vh',
				initialize: function() {

					// Deactivate section.
					if (skel.canUse('transition'))
						$(this).addClass('inactive');

				},
				enter: function() {

					// Activate section.
					$(this).removeClass('inactive');

				}
			})
			.each(function() {

				var $this = $(this),
					$image = $this.find('.image'),
					$img = $image.find('img'),
					x;

				// Assign image.
				$image.css('background-image', 'url(' + $img.attr('src') + ')');

				// Set background position.
				if (x = $img.data('position'))
					$image.css('background-position', x);

				// Hide <img>.
				$img.hide();

			});

		// Features.
		if (skel.canUse('transition'))
			$('.features')
			.scrollex({
				mode: 'middle',
				top: '-20vh',
				bottom: '-20vh',
				initialize: function() {

					// Deactivate section.
					$(this).addClass('inactive');

				},
				enter: function() {

					// Activate section.
					$(this).removeClass('inactive');

				}
			});

	});

})(jQuery);

/* Intro */
var canvas = document.querySelector("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
var ctx = canvas.getContext("2d");

var TAU = 2 * Math.PI;

times = [];

function loop() {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	update();
	draw();
	requestAnimationFrame(loop);
}

function Ball(startX, startY, startVelX, startVelY) {
	this.x = startX || Math.random() * canvas.width;
	this.y = startY || Math.random() * canvas.height;
	this.vel = {
		x: startVelX || Math.random() * 2 - 1,
		y: startVelY || Math.random() * 2 - 1
	};
	this.update = function(canvas) {
		if (this.x > canvas.width + 50 || this.x < -50) {
			this.vel.x = -this.vel.x;
		}
		if (this.y > canvas.height + 50 || this.y < -50) {
			this.vel.y = -this.vel.y;
		}
		this.x += this.vel.x;
		this.y += this.vel.y;
	};
	this.draw = function(ctx, can) {
		ctx.beginPath();
		ctx.globalAlpha = .4;
		ctx.fillStyle = '#448fda';
		ctx.arc((0.5 + this.x) | 0, (0.5 + this.y) | 0, 3, 0, TAU, false);
		ctx.fill();
	}
}

var balls = [];
for (var i = 0; i < canvas.width * canvas.height / (65 * 65); i++) {
	balls.push(new Ball(Math.random() * canvas.width, Math.random() * canvas.height));
}

var lastTime = Date.now();

function update() {
	var diff = Date.now() - lastTime;
	for (var frame = 0; frame * 36.6667 < diff; frame++) {
		for (var index = 0; index < balls.length; index++) {
			balls[index].update(canvas);
		}
	}
	lastTime = Date.now();
}
var mouseX = -1e9,
	mouseY = -1e9;
document.addEventListener('mousemove', function(event) {
	mouseX = event.clientX;
	mouseY = event.clientY;
});

function distMouse(ball) {
	return Math.hypot(ball.x - mouseX, ball.y - mouseY);
}

function draw() {
	ctx.globalAlpha = 0.8;
	ctx.fillStyle = '#001c33';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
	for (var index = 0; index < balls.length; index++) {
		var ball = balls[index];
		ball.draw(ctx, canvas);
		ctx.beginPath();
		for (var index2 = balls.length - 1; index2 > index; index2 += -1) {
			var ball2 = balls[index2];
			var dist = Math.hypot(ball.x - ball2.x, ball.y - ball2.y);
			if (dist < 100) {
				ctx.strokeStyle = "#2967a5";
				ctx.globalAlpha = 1 - (dist > 100 ? .8 : dist / 150);
				ctx.lineWidth = "2px";
				ctx.moveTo((0.5 + ball.x) | 0, (0.5 + ball.y) | 0);
				ctx.lineTo((0.5 + ball2.x) | 0, (0.5 + ball2.y) | 0);
			}
		}
		ctx.stroke();
	}
}

// Start
loop();