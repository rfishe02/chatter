var express = require('express');
var router = express.Router();
const multer  = require('multer');

// TODO: Consider switching to disk storage allow processing of larger files.
const storage = multer.memoryStorage()
const upload = multer({ storage: storage, limits: { fileSize: 1000000 }})

const ffmpeg = require('fluent-ffmpeg');
const { Readable, Writable } = require('stream');

router.post('/', upload.single('recording'), async function(req, res, next) {

    const fileUpload = req.file;
    if (!fileUpload) {
        return res.status(400).send({'message':'No files received'});
    }


    const webmBuffer = fileUpload.buffer; // Buffer type

    // Create a readable stream from the in-memory data
    const inputStream = new Readable();
    inputStream.push(webmBuffer);
    inputStream.push(null); // End the stream

    // Create a FFmpeg command
    const command = ffmpeg()
    .input(inputStream)
    .inputFormat('webm')
    .audioCodec('pcm_s16le') // Specify audio codec for WAV
    .toFormat('wav')
    .on('end', () => {
        console.log('Conversion finished');
    })
    .on('error', (err) => {
        console.error('Error:', err);
        res.status(500).end('Error during conversion');
    });

    res.setHeader('Content-Type', 'audio/wav');
    res.setHeader(
        'Content-Disposition',
        'attachment; filename=converted.wav'
    );

    // Pipe the output stream to the response
    command.pipe(res, { end: true });

});

module.exports = router;