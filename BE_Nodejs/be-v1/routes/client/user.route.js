const express = require('express')
const router = express.Router()
const multer = require('multer');
const upload = multer({ storage: multer.memoryStorage() }); 
const controllers = require("../../controllers/client/user.controller")
const uploadImageToCloudinary = require("../../middlewares/uploadImageToCloudinary")
const validate = require("../../validates/client/user.validate.js")



router.post('/register',upload.single("avatar"),uploadImageToCloudinary,validate.register,controllers.register )
router.post("/login",validate.login,controllers.login)
router.post("/checkToken",controllers.checkToken)




module.exports = router;