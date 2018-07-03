var margin = { top: 50, right: 20, bottom: 30, left: 100 },
cellWidth=40,
cellHeight=10,
col_number=7,
row_number=24,

width = cellWidth * col_number, // - margin.left - margin.right,
height = cellHeight * row_number , // - margin.top - margin.bottom,

//gridSize = Math.floor(width / 24),
legendElementWidth = cellWidth/3,
colorBuckets = 21,
colors = ['rgba(5,240,230,0)',
'rgba(5,240,230,0.05)','rgba(5,240,230,0.1)','rgba(5,240,230,0.15)','rgba(5,240,230,0.2)','rgba(5,240,230,0.25)',
'rgba(5,240,230,0.3)','rgba(5,240,230,0.35)','rgba(5,240,230,0.4)','rgba(5,240,230,0.45)','rgba(5,240,230,0.5)',
'rgba(5,240,230,0.55)','rgba(5,240,230,0.6)','rgba(5,240,230,0.65)','rgba(5,240,230,0.7)','rgba(5,240,230,0.75)',
'rgba(5,240,230,0.8)','rgba(5,240,230,0.85)','rgba(5,240,230,0.9)','rgba(5,240,230,0.95)','rgba(5,240,230,1)'],
rowLabel = ['00:00 AM','01:00 AM','02:00 AM','03:00 AM','04:00 AM','05:00 AM','06:00 AM','07:00 AM','08:00 AM','09:00 AM','10:00 AM','11:00 AM',
'12:00 AM','01:00 PM','02:00 PM','03:00 PM','04:00 PM','05:00 PM','06:00 PM','07:00 PM','08:00 PM','09:00 PM','10:00 PM','11:00 PM'],
colLabel = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];

d3.tsv("data_heatmap.tsv",
    function(d) {
        return {
            row:   +d.row_idx,
            col:   +d.col_idx,
            value: +d.log2ratio
        };
    },
    function(error,data) {

        var colorScale = d3.scale.quantile()
        .domain([ 1000 , 3400, 7800])
        .range(colors);

        var svg = d3.select("#chart").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var rowSortOrder=false;
        var colSortOrder=false;

        var rowLabels = svg.append("g").selectAll(".rowLabelg")
        .data(rowLabel)
        .enter()
        .append("text")
        .text(function (d) { return d; })
        .attr("x", 0)
        .attr("y", function (d, i) { return i * cellHeight; })
        .style("text-anchor", "end")
        .attr("transform", "translate(-10," + cellHeight / 1.5 + ")")
        .attr("class", function (d,i) { return "rowLabel mono r"+i;} )
        .on("mouseover", function(d) {d3.select(this).classed("text-hover",true);})
        .on("mouseout" , function(d) {d3.select(this).classed("text-hover",false);});

        var colLabels = svg.append("g").selectAll(".colLabelg")
        .data(colLabel)
        .enter()
        .append("text")
        .text(function (d) { return d; })
                // Giocare col parametro 7 per centrare
                .attr("x", function (d, i) { return i * cellWidth + (cellWidth-(d.length*7))/2; })
                .attr("y", cellHeight*25.25)
                .style("text-anchor", "left")
                .attr("class",  function (d,i) { return "colLabel mono c"+i;} )
                .on("mouseover", function(d) {d3.select(this).classed("text-hover",true);})
                .on("mouseout" , function(d) {d3.select(this).classed("text-hover",false);});

                var heatMap = svg.append("g").attr("class","g3").selectAll(".cellg")
                .data(data, function(d){return d.row+":"+d.col;})
                .enter()
                .append("rect")
                .attr("x", function(d) { return (d.col-1) * cellWidth; })
                .attr("y", function(d) { return (d.row-1) * cellHeight; })
                .attr("class", function(d){return "cell cell-border cr"+(d.row-1)+" cc"+(d.col-1);}) // griglia
                .attr("width", cellWidth)
                .attr("height", cellHeight)
                .style("fill", function(d) { return colorScale(d.value); })
                .on("mouseover", function(d){
                    //highlight text
                    d3.select(this).classed("cell-hover",true);
                    d3.selectAll(".rowLabel").classed("text-highlight",function(r,ri){ return ri==(d.row-1);});
                    d3.selectAll(".colLabel").classed("text-highlight",function(c,ci){ return ci==(d.col-1);});

                    //Update the tooltip position and value
                    d3.select("#tooltip")
                    .style("left", (d3.event.pageX+10-310) + "px")
                    .style("top", (d3.event.pageY-10-1550) + "px")
                    .select("#value")
                    .text("Quantity: "+d.value);
                    //Show the tooltip
                    d3.select("#tooltip").classed("hidden", false);
                })
                .on("mouseout", function(){
                    d3.select(this).classed("cell-hover",false);
                    d3.selectAll(".rowLabel").classed("text-highlight",false);
                    d3.selectAll(".colLabel").classed("text-highlight",false);
                    d3.select("#tooltip").classed("hidden", true);
                })
                ;

                var legend = svg.selectAll(".legend")
                .data([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
                .enter()
                .append("g")
                .attr("class", "legend");

                legend.append("rect")
                .attr("x", function(d, i) { return legendElementWidth * i; })
                .attr("y", height+(cellHeight*2))
                .attr("width", legendElementWidth)
                .attr("height", cellHeight)
                .style("fill", function(d, i) { return colors[i]; });

        /*
        legend.append("text")
            .attr("class", "mono")
            .text(function(d) { return d; })
            .attr("width", legendElementWidth)
            .attr("x", function(d, i) { return legendElementWidth * i; })
            .attr("y", height + (cellHeight*4));*/

        });