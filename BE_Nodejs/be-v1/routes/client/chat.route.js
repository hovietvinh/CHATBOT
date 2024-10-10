const express = require('express')
const router = express.Router()
const controllers = require("../../controllers/client/chat.controller")
const middlewares = require("../../middlewares/checkAuthAdmin")


// router.get('/',controllers.index )
router.post('/create',middlewares.checkAuthByToken,controllers.create )
router.get('/:id',controllers.getById )
router.get('/',middlewares.checkAuthByToken,controllers.index )
// router.post('/update',controllers.update )
router.patch("/sendMess",middlewares.checkAuthByToken,controllers.sendMess)



module.exports = router;