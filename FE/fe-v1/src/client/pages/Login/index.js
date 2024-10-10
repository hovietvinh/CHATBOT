import { Form, Input } from 'antd';
import React, { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { PiUserCircle } from "react-icons/pi";
import {useDispatch,useSelector} from "react-redux"
import { userLoginAction, userSetAction } from '../../../redux/client/actions/UserAction';
import toast from "react-hot-toast"
import { userCheckTokenApi, userLoginApi } from '../../../utils/client/api';

function Login() {
    const [form] = Form.useForm()
    const dispatch = useDispatch()
    const stateUser = useSelector(state=>state.UserReducer)
    const navigate = useNavigate()
    const [token,setToken] = useState(localStorage.getItem("user_token"))

    const handleLogin = async(e)=>{
        // console.log(e);
        // const res = await dispatch(userLoginAction(e))
        const res = await userLoginApi(e)
        console.log(res);
        if(res.code==200){
            toast.success("Login successfully!")
           
            // console.log(stateUser);
            // localStorage.setItem("user_token",stateUser.user_token)
            // console.log(res);
            dispatch(userLoginAction(res))
        }
        else{
            toast.error("Data invalid!")
        }

    }

    const checkUSer = async()=>{


        if(stateUser.token){
            localStorage.setItem("user_token",stateUser.token)
            navigate("/")

        }
        if(token){
            // console.log(token);
            const res = await userCheckTokenApi({token:token});
            // console.log(res);
            if(res.code==200){
                dispatch(userSetAction(res))
                navigate("/")

            }
            else{
                dispatch({type:"USER_LOGOUT"})
                localStorage.removeItem("user_token");
                setToken("")
                
            }
        }
       
        // console.log(stateUser);
    }
    useEffect(()=>{
        checkUSer()
        // console.log("load");
    },[stateUser,token])


    return (
        <>
            <div className='mt-5'>
                <div className='bg-white w-full max-w-sm rounded overflow-hidden p-4 mx-auto'>
                    <div className='w-fit mx-auto mb-2'>
                        <PiUserCircle
                            size={80}
                        />
                    </div>
                    <h3>Welcome to Chat app!</h3>
                    {/* <button className='w-full font-semibold text-white text-[16px] bg-primary px-4 py-1 transition-all duration-300 rounded-md hover:bg-secondary ' htmltype="submit">Login</button> */}

                    <Form
                    layout='vertical'
                    className='mt-3'
                    form={form}
                    onFinish={handleLogin}
                    
                    >
                        <Form.Item name="email" label="Email :" className=''
                            rules={[{ required: true, message: 'Please fill in this field!' }, { type: 'email', message: 'Invalid email format!' },]}
                            
                        >
                            <Input 
                                placeholder='enter your name'
                                className='bg-slate-100 px-2 py-1'
                                
                            />

                        </Form.Item>

                        <Form.Item name="password" label="Password :" className=''
                            rules={[{ required: true, message: 'Please fill in this field!' }]}    
                        >
                            <Input.Password 
                                placeholder='enter your password'
                                className='bg-slate-100 px-2 py-1'
                            />

                        </Form.Item>

                        

                        <Form.Item>
                            <button className='w-full font-semibold text-white text-[16px] bg-primary px-4 py-1 transition-all duration-300 rounded-md hover:bg-secondary ' htmltype="submit">Login</button>
                        </Form.Item>

                        <p className='my-3 text-center'>New user ? <Link to={"/users/register"} className="transition duration-300 hover:text-primary font-medium ">Register</Link></p>
                        

                    </Form>
                </div>
            </div>        
        </>
    );
}

export default Login;