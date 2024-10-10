import { MenuOutlined, PlusOutlined, MessageOutlined, LogoutOutlined,QuestionCircleOutlined, SettingOutlined, ClockCircleOutlined } from "@ant-design/icons";
import { useEffect, useState } from "react";
import { toast } from "react-hot-toast";
import { useDispatch } from "react-redux";
import { useLocation, useNavigate } from "react-router-dom";
import { getAllChatApi } from "../../../utils/client/api";

function Menu({ collapsed, handleCollapsed }) {
    const naviagate = useNavigate()
    const location = useLocation()
    const dispatch = useDispatch()

    const [chats,setChats] = useState([])
    

    const fetch =async ()=>{
        const res = await getAllChatApi();
        if(res.code==200){
            setChats(res.data)
        }
    }
    useEffect(()=>{
        fetch()
    },[location])

    const shortText = (text) => {
        if (text.length > 18) {
            let short = text.substring(0, 15);
         
            const lastSpace = short.lastIndexOf(" ");
            if (lastSpace > 0) {
                short = short.substring(0, lastSpace);
            }
            return short + "...";
        } else {
            return text;
        }
    };


    const handleNewChat = ()=>{
        naviagate("/");
    }
    const changeTab =(id)=>{
        naviagate(`chat/${id}`);

    }
    const handleLogout = ()=>{
        dispatch({type:"USER_LOGOUT"});
        localStorage.removeItem("user_token");
        naviagate("/users/login");
        toast.success("Logout successfully!")

    }

    return (
        <>
            <div className="min-h-[100vh] inline-flex flex-col justify-between bg-[#f0f4f9] py-[30px] px-[20px] w-full animate-fadeIn">

                <div className="">
                    <MenuOutlined onClick={handleCollapsed} className="text-xl p-2 rounded-md mb-10 transition duration-300 hover:bg-slate-300 cursor-pointer mb-3" />
                    
                    <div onClick={handleNewChat} className="text-lg w-[80%] transition duration-300 hover:bg-slate-300 font-normal p-2 rounded-2xl bg-slate-200 cursor-pointer flex justify-center items-center gap-2">
                        <PlusOutlined  className=""/> {collapsed ? <p className=""></p> : <p className="">New chat</p>}
                    </div>

                    <div className="flex flex-col" >
                        {collapsed ? (
                            <div className="flex gap-3 mt-8 mb-5 animate-fadeIn"><ClockCircleOutlined /> </div>
                            
                        ) : (
                            <div className="flex gap-3 mt-8 mb-5 animate-fadeIn">
                                <ClockCircleOutlined /> <p className="text-black font-normal text-lg overflow-x-auto">Recent</p>
                            </div>
                        )}

                        <div className="h-[470px] overflow-y-auto">
                            {chats.map((chat, idx) => (
                                <div onClick={()=>{changeTab(chat._id)}} key={idx} className="flex p-2 items-center gap-2 transition duration-300 rounded-lg cursor-pointer hover:bg-slate-200">
                                    <MessageOutlined className="text-lg" /> {collapsed ? <p></p> : <p>{shortText(chat?.chatContent?.[0]?.chatUser)}</p>}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
                
                <div className="flex flex-col">
                    
                    <div className="flex p-2 items-center gap-2 transition duration-300 rounded-lg cursor-pointer hover:bg-slate-200">
                        <QuestionCircleOutlined className="pr-1" /> {collapsed ? <p></p> : <p>Help</p>}
                    </div>
                    <div className="flex p-2 items-center gap-2 transition duration-300 rounded-lg cursor-pointer hover:bg-slate-200">
                        <SettingOutlined className="pr-1" /> {collapsed ? <p></p> : <p>Settings</p>}
                    </div>
                    <div onClick={handleLogout} className="flex p-2 items-center gap-2 transition duration-300 rounded-lg cursor-pointer hover:bg-slate-200">
                        <LogoutOutlined className="pr-1" /> {collapsed ? <p></p> : <p>Đăng xuất</p>}
                    </div>
                </div>
            </div>
        </>
    );
}

export default Menu;
