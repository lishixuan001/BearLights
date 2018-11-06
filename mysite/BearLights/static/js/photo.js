var video, canvas, context, photo, vendorUrl;

var dataUrl, blob, arrayBuffer, data_stream;

(function() {
   video = document.getElementById('video');
   canvas = document.getElementById('canvas');
   context = canvas.getContext('2d');
   photo = document.getElementById('photo');
   vendorUrl = window.URL || window.webkitURL;

   navigator.getMedia = navigator.getUserMedia ||
       navigator.webkitGetUserMedia ||
       navigator.mozGetUserMedia ||
       navigator.msGetUserMedia;

   navigator.getMedia({
       video: true,
       audio: false
   }, function(stream) {
       video.src = vendorUrl.createObjectURL(stream);
       video.play();
   }, function (error) {
       // An error occured
       // error.code
   });

    document.getElementById('capture').addEventListener('click', function() {
        context.drawImage(video, 0, 0, 400, 300);
        dataUrl = canvas.toDataURL('image/png');
        photo.setAttribute('src', dataUrl);

        blob = dataURLtoBlob(dataUrl);

        var fileReader = new FileReader();
        fileReader.onload = function(event) {
            arrayBuffer = event.target.result;
        };
        fileReader.readAsArrayBuffer(blob);

        data_stream = new Uint8Array(arrayBuffer);

        setTimeout(submitPhoto, 2000);

        // var a = document.createElement('a');
        // a.href = dataUrl;
        // a.download = "myImage.png";
        // a.style.display = 'none';
        // document.body.appendChild(a);
        // a.click();

    });

})();

function submitPhoto() {
    $("#data").val(dataUrl);
    $("#submit").click();
}

function readBlobAsDataURL(blob, callback) {
    var a = new FileReader();
    a.onload = function(e) {
        callback(e.target.result);
    };
    a.readAsDataURL(blob);
}

function dataURLtoBlob(dataurl) {
    var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new Blob([u8arr], {type:mime});
}

function dataURLtoFile(dataurl, filename) {
    var arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
        bstr = atob(arr[1]), n = bstr.length, u8arr = new Uint8Array(n);
    while(n--){
        u8arr[n] = bstr.charCodeAt(n);
    }
    return new File([u8arr], filename, {type:mime});
}
