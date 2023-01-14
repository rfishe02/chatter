var express = require('express');
var router = express.Router();
const multer  = require('multer');

// TODO: Consider switching to disk storage allow processing of larger files.
const storage = multer.memoryStorage()
const upload = multer({ storage: storage, limits: { fileSize: 1000000 }})

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
    ffmpeg.FS('writeFile', 'recording.webm', webmBuffer);
    await ffmpeg.run('-i','recording.webm','-ar','16000','recording.wav'); // Conver to format compatible with HuggingFace models, wav & lower sampling rate.
    const wavArray = ffmpeg.FS('readFile', 'recording.wav'); // Uint8Array type
    const wavBuffer = Buffer.from(wavArray, 'binary'); // Need to wrap in Buffer to ensure Blob conversion in the browser.

    res.set('Content-Type', 'audio/wav');
    return res.status(200).send(wavBuffer); 

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