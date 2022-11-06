$(function() {
    $('#upload-file-btn').click(function() {
        var form_data = new FormData($('#upload-file')[0]);
        $.ajax({
            type: 'POST',
            url: '/uploadajax',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: false,
            success: function(data) {
                console.log('Success!');
            },
        });
    });
});




// upload multiple form using jquery
$(document).ready(function() {
    $("#sendAll").click(function() {
        // data diolah disini kemudian passing as json ke python
        var filename = $("input[name='filename']").val();
        var comment = $("textarea[name='message']").val();
        var photos = new FormData($("#imageform")[0]);
        // console.log(filename);
        // console.log(comment);
        // console.log(photos);
        photos.append("filename", filename);
        photos.append("comment", comment);

        // send via ajax this data
        $.ajax({
            url: "/upload_page",
            type: "POST",

            // progress handling start
            xhr: function() {  // Custom XMLHttpRequest
            var myXhr = $.ajaxSettings.xhr();
            if (myXhr.upload) { // Check if upload property exists
                myXhr.upload.addEventListener('progress',progressHandlingFunction, false); // For handling the progress of the upload
            } else {
                console.log("Upload progress is not supported!");
            }
            return myXhr;
            },

            // ajax events
            // beforeSend: beforeSendHandler,
            // success: completeHandler,
            // error: errorHandler,
            // progress handling ends

            data: photos,
            cache: false,
            // async: false,
            processData: false,
            contentType: false,
            success: function(response) {
                // console.log(response);
                // window.location.reload(true);
                console.log("hai")
                // harusnya clear semua form
                $("input[name='filename']").val("");
                $("textarea[name='message']").val("");
                $(":file").val("");
                $("progress").attr({"value": 0, "total": 100});
            }
        });
    });
});





function progressHandlingFunction(e) {
    if (e.lengthComputable) {
        console.log("hai progress!");
        console.log(e.loaded);
        console.log(e.total);
        $("progress").attr({value: e.loaded, max: e.total});
    }
}