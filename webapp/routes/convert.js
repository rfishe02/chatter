var express = require('express');
var router = express.Router();
const multer  = require('multer');

// TODO: Consider switching to disk storage allow processing of larger files.
const storage = multer.memoryStorage()
const upload = multer({ storage: storage, limits: { fileSize: 1000000 }})

var ffmpegLibrary = require('@ffmpeg/ffmpeg');

router.post('/', upload.single('recording'), async function(req, res, next) {

    const fileUpload = req.file;
    if (!fileUpload) {
        return res.status(400).send({'message':'No files received'});
    }

    const ffmpeg = ffmpegLibrary.createFFmpeg({ log: true });
    if(!ffmpeg.isLoaded()) {
        await ffmpeg.load();
    }

    const webmBuffer = fileUpload.buffer; // Buffer type
    ffmpeg.FS('writeFile', 'recording.webm', Uint8Array.from(webmBuffer));
    await ffmpeg.run('-i','recording.webm','-ar','16000','recording.wav'); // Conver to format compatible with HuggingFace models, wav & lower sampling rate.
    const wavArray = ffmpeg.FS('readFile', 'recording.wav'); // Uint8Array type
    const wavBuffer = Buffer.from(wavArray, 'binary'); // Need to wrap in Buffer to ensure Blob conversion in the browser.

    ffmpeg.exit(); // Possibly avoid Uncaught RuntimeError: abort(OOM)

    res.set('Content-Type', 'audio/wav');
    return res.status(200).send(wavBuffer); 

});

module.exports = router;