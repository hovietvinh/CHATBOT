import { userLoginApi } from "../../../utils/client/api"
import toast from 'react-hot-toast';

const userLoginAction = (res)=>{
    // return async(dispatch)=>{
    //     try {
    //         const res = await userLoginApi(data)
        
    //         if(res.code==200){
    //             dispatch({
    //                 type:"USER_LOGIN",
    //                 data:res.data,
    //                 user_token:res.user_token
    //             })
    //             return true
    //         }
    //         else{
    //             return false 
    //         }
    //     } catch (error) {
    //         return false
    //     }
    // }

    return {
        type:"USER_LOGIN",
        data:res.data,
        user_token:res.user_token
    }
}

const userSetAction = (data)=>{
    return {
        type:"USER_SET",
        data:data.data,
        user_token:data.user_token
    }
}


export {
    userLoginAction,
    userSetAction
}