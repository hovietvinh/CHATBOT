const chatRouter = require("./chat.route");
const userRouter = require("./user.route");
module.exports = (app)=>{
    // app.get('/', (req, res) => {
    //     res.send('Hello Worladd!')
    // })

    app.use('/api/chat', chatRouter)
    app.use("/api/users",userRouter)

}