var express = require('express');
var router = express.Router();
const multer  = require('multer');

// We're using in memory storage to lessen conversion time. If the system recives large files, this could be an issue.
const storage = multer.memoryStorage()
const upload = multer({ storage: storage })

var ffmpegLibrary = require('@ffmpeg/ffmpeg');
const ffmpeg = ffmpegLibrary.createFFmpeg({ log: true });

router.post('/', upload.single('recording'), async function(req, res, next) {

    const fileUpload = req.file;

    if (!fileUpload) {
        return res.status(400).send({'message':'No files received'});
    }

    if(!ffmpeg.isLoaded()) {
        await ffmpeg.load();
    }

    const webmBuffer = fileUpload.buffer; // Buffer type
    ffmpeg.FS('writeFile', './recording.webm', webmBuffer);
    await ffmpeg.run('-i', './recording.webm', './recording.mp3');
    const mp3Array = ffmpeg.FS('readFile', './recording.mp3'); // Uint8Array type
    const mp3Buffer = Buffer.from(mp3Array, 'binary'); // Need to wrap in Buffer to ensure Blob conversion in the browser.

    res.set('Content-Type', 'audio/mp3');
    return res.status(200).send(mp3Buffer); 

});

// May not be necessary to use since Buffer might be backed by a Uint8Array.
function toUint8Array(buffer) {
    const view = new Uint8Array(buffer.length);
    for (let i = 0; i < buffer.length; i++) {
        view[i] = buffer[i];
    }
    return view;
}

module.exports = router;