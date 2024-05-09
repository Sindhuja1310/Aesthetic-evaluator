/* script.js */

$(document).ready(function() {
    // Show the upload option
    $("#uploadOption").click(function() {
        $("#uploadSection").show();
        $("#captureSection").hide();
        $("#imagePreviewContainer").hide(); // Hide image preview container
        $("#evaluateButton").hide(); // Hide evaluate button
        $("#result").hide(); // Hide result section
    });

    // Show the capture option
    $("#captureOption").click(function() {
        $("#captureSection").show();
        $("#uploadSection").hide();
        $("#imagePreviewContainer").hide(); // Hide image preview container
        $("#evaluateButton").hide(); // Hide evaluate button
        $("#result").hide(); // Hide result section

        // Access the device's camera
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    var video = document.getElementById('video');
                    video.srcObject = stream;
                    video.play();
                    $("#captureButton").show(); // Show the capture button
                    $("#canvas").show(); // Show the canvas for live preview
                    
                    // Add live preview functionality
                    var canvas = document.getElementById('canvas');
                    var context = canvas.getContext('2d');
                    setInterval(function() {
                        context.drawImage(video, 0, 0, canvas.width, canvas.height);
                    }, 100); // Update every 100 milliseconds
                })
                .catch(function(error) {
                    console.error('Error accessing camera: ', error);
                });
        }
    });

    // Capture picture from the camera
    $("#captureButton").click(function() {
        var canvas = document.getElementById('canvas');
        var context = canvas.getContext('2d');
        var video = document.getElementById('video');

        // Draw the video frame on the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        console.log(canvas.toDataURL('image/png'));

        // Show the captured image
        $("#imagePreviewContainer").show(); // Show image preview container
        $("#previewImage").attr('src', canvas.toDataURL('image/png'));

        // Stop video stream
        video.srcObject.getVideoTracks().forEach(track => track.stop());

        // Hide live preview and canvas
        $("#video").hide();
        $("#canvas").hide();

        // Show the evaluate button
        $("#evaluateButton").show();
    });

    // Upload image from the device
    $("#uploadButton").click(function() {
        $("#imageInput").click();
    });

    $("#imageInput").change(function() {
        readURL(this);
    });

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();

            reader.onload = function(e) {
                $('#imagePreviewContainer').show(); // Show image preview container
                $("#imagePreviewContainer").css("display", "block"); // Set image preview container to display as "block"
                $('#previewImage').attr('src', e.target.result);
                console.log(e.target.result);
                $("#evaluateButton").show(); // Show the evaluate button
            }

            reader.readAsDataURL(input.files[0]);
        }
    }

    // Evaluate the image
    $("#evaluateButton").click(function() {
        var imgData = document.getElementById("previewImage").src;
        console.log(imgData);
        $.ajax({
            type: "POST",
            url: "/evaluate",
            data: { image_data: imgData },
            success: function(response) {
                console.log(response);
                $("#balanceScore").text(response.balance_score.toFixed(2));
                $("#proportionScore").text(response.proportion_score.toFixed(2));
                $("#symmetryScore").text(response.symmetry_score.toFixed(2));
                $("#simplicityScore").text(response.simplicity_score.toFixed(2));
                $("#harmonyScore").text(response.harmony_score.toFixed(2));
                $("#contrastScore").text(response.contrast_score.toFixed(2));
                $("#unityScore").text(response.unity_score.toFixed(2));
                $("#averageAestheticValue").text(response.average_aesthetic_value.toFixed(2));

                // Show the result section
                $("#result").show();
            },
            error: function(xhr, status, error) {
                console.error('Error occurred while evaluating image');
            }
        });
    });
});
