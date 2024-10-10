const cloudinary = require('cloudinary').v2;
const multer = require('multer');
const streamifier = require('streamifier');

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET
});
const uploadImageToCloudinary = async (req, res, next) => {
    // console.log(req.file);
    if (!req.file) {
        // console.log('No file uploaded');
        next(); 
        return ; // Continue to next middleware without image
    }
  
    try {
        
        const streamUpload = (file) => {
            return new Promise((resolve, reject) => {
              const stream = cloudinary.uploader.upload_stream((error, result) => {
                if (result) {
                  resolve(result);
                } else {
                  reject(error);
                }
              });
              streamifier.createReadStream(file.buffer).pipe(stream);
            });
          };
      
      
      const result = await streamUpload(req.file);
      
      req.body.thumbnail = result.secure_url;  // Set Cloudinary URL
      next();
    } catch (error) {
    //   res.status(500).send({ message: 'Cloudinary Upload Error', error });
        console.log(req.body,2);
    }
  };
    

module.exports = uploadImageToCloudinary;
