$( document ).ready(function() {
    
    var ctx6 = document.getElementById("chart6").getContext("2d");
    var data6 = {
        labels: ["Internet", "Reseau", "Offres", "Appelle", "Autres"],
        datasets: [
            {
                label: "Negative",
                fillColor: " #ee9998",
                strokeColor: "#ea100c",
                pointColor: "#ea100c",
                pointStrokeColor: "#fff",
                pointHighlightFill: "#fff",
                pointHighlightStroke: "#ea100c",
                data: ["{{internet_count}}", "{{reseau_count}}", "{{offre_count}}", "{{appelle_count}}", "{{other_count}}"]
            },
            {
                label: "Neutre",
                fillColor: "rgba( 164, 196, 245, .7)",
                strokeColor: "#4184eb",
                pointColor: "#4184eb",
                pointStrokeColor: "#fff",
                pointHighlightFill: "#fff",
                pointHighlightStroke: "#4184eb",
                data: [28, 48, 40, 19, 96]
            }, 
            {
                label: "Positive",
                fillColor: "rgba(161, 234, 134, .7)",
                strokeColor: "#44cf10",
                pointColor: "#44cf10",
                pointStrokeColor: "#fff",
                pointHighlightFill: "#fff",
                pointHighlightStroke: "#44cf10",
                data: [5, 39, 80, 41, 66]
            },
        ]
    };
    
    var myRadarChart = new Chart(ctx6).Radar(data6, {
        scaleShowLine : true,
        angleShowLineOut : true,
        scaleShowLabels : false,
        scaleBeginAtZero : true,
        angleLineColor : "rgba(0,0,0,.1)",
        angleLineWidth : 1,
        pointLabelFontFamily : "'Arial'",
        pointLabelFontStyle : "normal",
        pointLabelFontSize : 10,
        pointLabelFontColor : "#666",
        pointDot : true,
        pointDotRadius : 3,
		tooltipCornerRadius: 2,
        pointDotStrokeWidth : 1,
        pointHitDetectionRadius : 20,
        datasetStroke : true,
        datasetStrokeWidth : 2,
        datasetFill : true,
        legendTemplate : "<ul class=\"<%=name.toLowerCase()%>-legend\"><% for (var i=0; i<datasets.length; i++){%><li><span style=\"background-color:<%=datasets[i].strokeColor%>\"></span><%if(datasets[i].label){%><%=datasets[i].label%><%}%></li><%}%></ul>",
        responsive: true
    });
    
});