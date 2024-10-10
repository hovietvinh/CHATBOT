require("dotenv").config()
const jwt = require("jsonwebtoken")
module.exports.checkAuthByToken = (req, res, next) => {
    if(req.headers && req.headers.authorization){
        const token = req.headers.authorization.split(" ")[1];
        try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET)
            req.user = decoded
            // console.log("thanh cong");
            next()
        } catch (error) {
            return res.json({
                code:400,
                message:"Không có quyền truy cập"
            })
        }
        
    }
    else{
        return res.json({
            code:400,
            message:"Không có quyền truy cập"
        })
    }
    
};