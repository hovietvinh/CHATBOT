import axios from "../axios.customize"

const userRegisterApi = async(data)=>{
    try {
        
        const URL_LOGIN ='/api/users/register'
        const response = await axios.post(URL_LOGIN,data)
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}

const userLoginApi = async(data)=>{
    try {
        const URL_LOGIN ='/api/users/login'
        const response = await axios.post(URL_LOGIN,data,{
            headers:{
                "Content-Type":"application/json"
            }
        })
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}

const userCheckTokenApi = async(token)=>{
    try {
        const URL_LOGIN ='/api/users/checkToken'
        const response = await axios.post(URL_LOGIN,token,{
            headers:{
                "Content-Type":"application/json"
            }
        })
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}


const createChatApi = async(data)=>{
    try {
        const URL_LOGIN ='/api/chat/create'
        const response = await axios.post(URL_LOGIN,data,{
            headers:{
                "Content-Type":"application/json"
            }
        })
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}

const getChatApi = async (id)=>{
    try {
        const URL_LOGIN =`/api/chat/${id}`
        const response = await axios.get(URL_LOGIN)
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}


const getAllChatApi = async (id)=>{
    try {
        const URL_LOGIN =`/api/chat`
        const response = await axios.get(URL_LOGIN)
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}

const sendMessApi = async (data)=>{
    try {
        
        const URL_LOGIN =`/api/chat/sendMess`
        const response = await axios.patch(URL_LOGIN,data,{
            headers:{
                "Content-Type":"application/json"
            }
        })
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}



export {
    userRegisterApi,
    userLoginApi,
    userCheckTokenApi,
    createChatApi,
    getChatApi,
    sendMessApi,
    getAllChatApi
}