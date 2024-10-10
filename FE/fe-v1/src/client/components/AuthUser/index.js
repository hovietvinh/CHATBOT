import React, { useEffect, useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { Outlet, useLocation, useNavigate } from 'react-router-dom';
import { userSetAction } from '../../../redux/client/actions/UserAction';
import { userCheckTokenApi } from '../../../utils/client/api';
import toast from 'react-hot-toast'
import Menu from '../Menu';

function AuthUser() {

    const [token,setToken] = useState(localStorage.getItem("user_token"));
    const navigate = useNavigate();
    const location = useLocation()
    const stateAuth = useSelector(state=>state.UserReducer)
    const dispatch = useDispatch()
    const [show,setShow] = useState(false);

    const checkToken = async()=>{
        const res = await userCheckTokenApi({token:token})
        if(res.code==200){
            if(res.user_token != stateAuth.token){
                dispatch(userSetAction(res))
                setShow(true)
                
            }
        }
        else{
            // toast.error("Please login!")
            setShow(false)
            navigate("/users/login")
        }


        if(stateAuth.token){
            setShow(true)
        }
    }

    // console.log(12);

    useEffect(()=>{
        checkToken()
    },[token,dispatch,location])

    const [collapsed,setCollapsed] = useState(false);

    const handleCollapsed = ()=>{
        setCollapsed(!collapsed);
    }


    return (
        <>
            {show && (
                <div className={`grid ${collapsed ? 'grid-cols-[80px,1fr]' : 'grid-cols-[280px,1fr]'} transition-all duration-300`}>

                    <section >
                        <Menu collapsed={collapsed} handleCollapsed={handleCollapsed} />
                    </section>

                    <section>
                        <Outlet/>
                    </section>
                </div>
            )}
            
            
        
        </>
    );
}

export default AuthUser;