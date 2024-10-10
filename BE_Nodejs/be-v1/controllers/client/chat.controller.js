const Chat = require("../../models/chat.model");
const { getResponse } = require("../../utils/api");
// const { createChat, getChatById, addMessage, deleteMessage, getAllChat } = require("../../services/chatServices")
//[GET] api/ 
module.exports.index = async (req, res) => {
    try {
        const find = {
            deleted:false,
            userId: req.user._id
        }

        const chats = await Chat.find(find);
        res.json({
            code:200,
            data:chats
        })
    } catch (error) {
        res.json({
            code:400,
            message:"Error in BE!"
        })
    }
}


//[POST] api/chat/create 
module.exports.create = async(req, res) => {
    try {
        // const {message} = req.body
        const response =  await getResponse(req.body)
        const chatData = new Chat({
            userId: req.user._id,
            chatContent:[{
                chatUser:req.body.message,
                chatMachine:response.response
            }]
        })
        await chatData.save()
        
        res.json({
            code:200,
            data:chatData
        })
    } catch (error) {
        res.json({
            code:400,
            message:"Error in BE!"
        })
    }
    

   
}
// //[GET] api/chat/:id 
module.exports.getById = async (req, res) => {
    try {
        const find = {
            deleted:false,
            _id:req.params.id
        }

        const chat = await Chat.findOne(find);
        res.json({
            code:200,
            data:chat
        })
    } catch (error) {
        res.json({
            code:400,
            message:"Error in BE!"
        })
    }
    // console.log(req.params.id);
    // res.send("!23")
    // const result = await getChatById(req.params.id)
    // return res.status(200).json({result})
}

// //[PATCH] api/chat/sendMess
module.exports.sendMess = async (req, res) => {
    const {message,_id} = req.body
    const response =  await getResponse(req.body)

    const newChatMessage = {
        chatUser:message,
        chatMachine:response.response
    }

    await Chat.updateOne({ _id: _id},{
         $push: { chatContent: newChatMessage } , 
    })
    
    return res.json({code:200})
}

// //[PATCH] api/chat/delete
// module.exports.delete = async (req, res) => {
  
    
//     const result = await deleteMessage(req.query.id)
//     return res.status(200).json({result})
// }
