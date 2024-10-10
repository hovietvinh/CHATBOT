const axios =require( "./axiosCustome")

module.exports.getResponse = async(data)=>{
    try {
        const URL_LOGIN ='/api/chat'
        // console.log(URL_LOGIN);
        const response = await axios.post(URL_LOGIN,data,{
            headers:{
                'Content-Type': 'application/json'
            }
        })
        // console.log(response);
        return response
    } catch (error) {
        return {
            code:400,
            message:"Error in axios!"
        }
    }
}

// module.exports.getResponse= instance;

