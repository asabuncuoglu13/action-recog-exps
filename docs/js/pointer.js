var URL = window.URL;
var cvs = document.getElementById('canvas');
var ctx = cvs.getContext('2d');
var res = document.getElementById('results');

cvs.addEventListener('mousemove', mousePos, false);

window.onload = function() {
    var inputImage = document.getElementById('inputImage');
    inputImage.addEventListener('change', handleImageFiles, false);
}

function mousePos(evt) {
    var rect = cvs.getBoundingClientRect();
    var x = parseInt(evt.clientX - rect.left);
    var y = parseInt(evt.clientY - rect.top);
    var p = ctx.getImageData(x, y, 1, 1).data;
    results.innerHTML = '<table style="width:100%;table-layout:fixed"><td>X: ' +
        x + '</td><td>Y: ' + y + '</td><td>Red: ' +
        p[0] + '</td><td>Green: ' + p[1] + '</td><td>Blue: ' +
        p[2] + '</td><td>Alpha: ' + p[3] + "</td></table>";
    return { x, y };
}

function handleImageFiles(e) {
    var url = URL.createObjectURL(e.target.files[0]);
    var img = new Image();
    img.onload = function() {
        cvs.width = img.width;
        cvs.height = img.height;
        ctx.drawImage(img, 0, 0);
    }
    img.src = url;
}