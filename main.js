// Mount the Image Slider
var splide = new Splide('#splide', {
    perPage: 1,
    rewind: false,
}).mount();

// Video Selector Input
document.querySelector('input').addEventListener('change', extractFrames, false);

function extractFrames() {
    var video = document.createElement('video');
    var array = [];
    var canvas = document.createElement('canvas');
    var ctx = canvas.getContext('2d');
    var pro = document.querySelector('#progress');

    function initCanvas(e) {
        canvas.width = this.videoWidth;
        canvas.height = this.videoHeight;
    }

    function mousePos(evt) {
        var rect = canvas.getBoundingClientRect();
        var x = parseInt(evt.clientX - rect.left);
        var y = parseInt(evt.clientY - rect.top);
        ctx = canvas.getContext('2d');
        var p = ctx.getImageData(x, y, 1, 1).data;
        results.innerHTML = '<table style="width:100%;table-layout:fixed"><td>X: ' +
            x + '</td><td>Y: ' + y + '</td><td>Red: ' +
            p[0] + '</td><td>Green: ' + p[1] + '</td><td>Blue: ' +
            p[2] + '</td><td>Alpha: ' + p[3] + "</td></table>";
        return { x, y };
    }

    function drawFrame(e) {
        this.pause();
        ctx.drawImage(this, 0, 0);
        canvas.toBlob(saveFrame, 'image/jpeg');
        pro.innerHTML = ((this.currentTime / this.duration) * 100).toFixed(2) + ' %';
        if (this.currentTime < this.duration) {
            this.play();
        }
    }

    function saveFrame(blob) {
        array.push(blob);
    }

    function revokeURL(e) {
        URL.revokeObjectURL(this.src);
    }

    function onend(e) {
        var img;
        // do whatever with the frames
        for (var i = 0; i < array.length; i += 500) {
            //document.getElementById("content").appendChild(img);
            img = new Image();
            img.onload = revokeURL;
            img.src = URL.createObjectURL(array[i]);
            img.addEventListener('mousemove', mousePos, false);
            var liItem = document.createElement('li');
            liItem.setAttribute('class', "splide__slide");
            liItem.appendChild(img);
            splide.add(liItem);
        }
        // we don't need the video's objectURL anymore
        URL.revokeObjectURL(this.src);
        splide.refresh();
    }

    video.muted = true;

    video.addEventListener('loadedmetadata', initCanvas, false);
    video.addEventListener('timeupdate', drawFrame, false);
    video.addEventListener('ended', onend, false);

    video.src = URL.createObjectURL(this.files[0]);
    video.play();
}