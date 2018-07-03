var processed_json = Array();	
$.getJSON(
    'timeline.json',
    function (data) {
		// Preprocessing the json
		for (i = 0; i < data.length; i++){
        	processed_json.push([Date.parse(data[i].date), data[i].quantity]);
        }
        
        Highcharts.chart('container', {
            chart: {
                zoomType: 'x',
                backgroundColor: 'rgba(255, 255, 255, 0)'
            },
            navigation: {
                buttonOptions: {
                    theme: {
                        fill: 'transparent',
                        states: {
                            hover: {
                                fill: '#FFFFFF'
                            },
                            select: {
                                fill: '#FFFFFF'
                            }
                        }
                    }
                }
            },
            title: {
                text: 'Distribution of the Tweets over time'
            },
            subtitle: {
                text: document.ontouchstart === undefined ?
                        'Click and drag in the plot area to zoom in' : 'Pinch the chart to zoom in'
            },
            credits: {
        		enabled: false
    		},
            xAxis: {
                type: 'datetime'
            },
            yAxis: {
                title: {
                    text: 'Number of Tweets'
                }
            },
            legend: {
                enabled: false
            },
            plotOptions: {
                area: {
                    fillColor: {
                        linearGradient: {
                            x1: 0,
                            y1: 0,
                            x2: 0,
                            y2: 1
                        },
                        stops: [
                            [0, Highcharts.getOptions().colors[0]],
                            [1, Highcharts.Color(Highcharts.getOptions().colors[0]).setOpacity(0).get('rgba')]
                        ]
                    },
                    marker: {
                        radius: 2
                    },
                    lineWidth: 1,
                    states: {
                        hover: {
                            lineWidth: 1
                        }
                    },
                    threshold: 0
                }
            },

            series: [{
                type: 'area',
                name: '#Tweets',
                data: processed_json
            }]
        });
    }
);