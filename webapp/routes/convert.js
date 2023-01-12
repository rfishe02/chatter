var express = require('express');
var router = express.Router();
const multer  = require('multer')

const storage = multer.memoryStorage()
const upload = multer({ storage: storage })

var ffmpegLibrary = require('@ffmpeg/ffmpeg');
const ffmpeg = ffmpegLibrary.createFFmpeg({ log: true });

router.post('/', upload.single('recording'), async function(req, res, next) {

    const fileUpload = req.file;

    if (!fileUpload) {
        return res.status(400).send({'message':'No files received'});
    }

    const fileBuffer = fileUpload.buffer;

    await ffmpeg.load();
    ffmpeg.FS('writeFile', 'recording.webm', fileBuffer);
    //await ffmpeg.run('-i', 'recording.webm', 'recording.mp3');
    //await fs.promises.writeFile('./recording.mp3', ffmpeg.FS('readFile', 'recording.mp3'));

    return res.status(200).send({'message':'Received POST'});

});

module.exports = router;