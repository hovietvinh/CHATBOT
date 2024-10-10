const mongoose = require("mongoose")
const chatSchema = new mongoose.Schema(
    { 
        userId:String,
        chatContent: {
            type: [
              {
                chatUser:String,
                chatMachine:String
              }
            ],
            default: [] 
          },
        deleted: {
            type:Boolean,
            default:false
        },
        deletedAt:Date,    
    }
 );

const Chat = mongoose.model('Chat', chatSchema,"chats");

module.exports = Chat

// chat [ ]