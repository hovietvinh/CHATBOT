import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";
import { userSetAction } from "../../../redux/client/actions/UserAction";
import { userCheckTokenApi } from "../../../utils/client/api";
import toast from "react-hot-toast"


function AuthLayout({children}) {
    const dispatch = useDispatch()
    const navigate = useNavigate()
    const location = useLocation()

    const [token,setToken] = useState(localStorage.getItem("user_token"));
    const checkUSer = async()=>{

        if(token){
            // console.log(token);
            const res = await userCheckTokenApi({token:token});
            // console.log(res);
            if(res.code==200){
                dispatch(userSetAction(res))
                
                // navigate("/")
            }
            else{
                dispatch({type:"USER_LOGOUT"})
                localStorage.removeItem("user_token");
                setToken("")
                toast.error("user_token invalid!")
                if(location.pathname =="/users/login" || location.pathname =="/users/register"){
                    navigate(location.pathname)
                }
                

                
            }
        }
       
        // console.log(stateUser);
    }
    useEffect(()=>{
        checkUSer()
    },[token,location])


    return (
        <>
            <header className="flex justify-center items-center py-3 h-20 shadow-md bg-white">
                {/* <h1 className="text-[30px] font-semibold text-primary">Chat App</h1>
                 */}
                <h1 className="text-[30px] font-semibold text-primary">Chat App</h1>

            </header>
        
            {children}
        </>
    );
}

export default AuthLayout;