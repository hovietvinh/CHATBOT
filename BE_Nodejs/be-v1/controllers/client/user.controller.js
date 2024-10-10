require("dotenv").config()
const User = require("../../models/user.model")
const md5 = require("md5")
const jwt = require("jsonwebtoken")

//[POST] api/user/register 
module.exports.register = async (req, res) => {
    try {

        // const data = req.body
        const existsUser = await User.findOne({
            deleted:false,
            email:req.body.email
        })
        if(existsUser){
            return res.json({
                code:400,
                message:"Already user exits"          
            })
        }
        // console.log(1);
        req.body.password = md5(req.body.password);
        
        const user = new User(req.body)
        user.save()
        // console.log(data);
        res.json({
            code:200,
            data:user,
            message:"User created successfully"          
        })
    } catch (error) {
        res.json({
            code:400,
            message:"Error in BE"
        })
    }   
}



//[POST] api/user/login 
module.exports.login = async (req, res) => {
    try {

        // const data = req.body
        const existsUser = await User.findOne({
            deleted:false,
            email:req.body.email
        })
        if(existsUser){
            req.body.password = md5(req.body.password);
            if(req.body.password=== existsUser.password){

                // const user_token = jwt.sign()
                const payload = {
                    _id:existsUser._id,
                    fullName:existsUser.fullName,
                    email:existsUser.email,
                    avatar:existsUser.avatar
                }

                const user_token = jwt.sign(
                    payload,
                    process.env.JWT_SECRET,
                    {
                        expiresIn: process.env.JWT_EXPIRE   
                    }
                )
                res.json({
                    code:200,
                    data:payload,
                    user_token
                })
            }
            else{
                return res.json({
                    code:400,
                    message:"Data invalid!"
                })
            }
        }
        else{
            return res.json({
                code:400,
                message:"Data invalid!"
            })
        }
        
        // console.log(1);
       
        
       
    } catch (error) {
        res.json({
            code:400,
            message:"Error in BE"
        })
    }   
}




//[POST] api/user/checkToken 
module.exports.checkToken = async (req, res) => {
    try {
        
        const {token} = req.body
        // console.log(token);


        try {
            const decoded = jwt.verify(token, process.env.JWT_SECRET)
            res.json({
                code:200,
                data:decoded,
                user_token:token  
            })
            
        } catch (error) {
            return res.json({
                code:400,
                message:"Không có quyền truy cập"
            })
        }
        
        // console.log(data);
       
    } catch (error) {
        res.json({
            code:400,
            message:"Error in BE"
        })
    }   
}

