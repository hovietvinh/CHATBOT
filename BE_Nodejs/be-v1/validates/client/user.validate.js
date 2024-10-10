module.exports.register = (req,res,next)=>{
    // console.log(req.body);
    if(!req.body.fullName){
        res.json({
            code:400,
            message:"Please fill in this field!"
        })
        return;
    }
    if(!req.body.email){
        res.json({
            code:400,
            message:"Please fill in this field!"
        })
        return;
    }
    if(!req.body.password){
        res.json({
            code:400,
            message:"Please fill in this field!"
        })
        return;
    }
  
    next()
}   

module.exports.login = (req,res,next)=>{
    // console.log(req.body);
   
    if(!req.body.email){
        res.json({
            code:400,
            message:"Please fill in this field!"
        })
        return;
    }
    if(!req.body.password){
        res.json({
            code:400,
            message:"Please fill in this field!"
        })
        return;
    }
  
    next()
}   

