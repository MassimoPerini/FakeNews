
Highcharts.chart('container4', {
    chart: {
        type: 'column',
        options3d: {
            enabled: true,
            alpha: 15,
            beta: 15,
            viewDistance: 25,
            depth: 40
        },
        backgroundColor: 'rgba(255, 255, 255, 0)'
    },

    title: {
        text: 'Dataset insights'
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

    subtitle: {
        //text: 'text'
    },

    xAxis: {
        categories: ['Tweets', 'News', 'Users'],
        labels: {
            skew3d: true,
            style: {
                fontSize: '16px'
            }
        }
    },

    yAxis: {
        allowDecimals: false,
        min: 0,
        title: {
            text: 'Quantity',
            skew3d: true
        }
    },

    credits: {
        enabled: false
    },

    tooltip: {
        headerFormat: '<b>{point.key}</b><br>',
        //pointFormat: '<span style="color:{series.color}">\u25CF</span> {series.name}: {point.y} / {point.stackTotal}'
        pointFormat: '<span style="color:{series.color}">\u25CF</span> {series.name}: {point.y} %'
    },

    plotOptions: {
        column: {
            stacking: 'normal',
            depth: 40
        }
    },

    series: [{
        name: 'Real',

        data: [{y: 72, color: 'rgb(34, 205, 232)'},
                {y: 49, color: 'rgb(34, 205, 232)'},
                {y: 33, color: 'rgb(34, 205, 232)'}],
        stack: 'real'
    },  {
        name: 'Fake',
        data: [{y: 28, color: 'rgb(21,64,71)'},
            {y: 51, color: 'rgb(21,64,71)'},
            {y: 67, color: 'rgb(21,64,71)'}],
        stack: 'fake'
    }]
});