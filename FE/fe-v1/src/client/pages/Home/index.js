import { Form, Image, Input, Skeleton } from 'antd';
import React, { useEffect, useState } from 'react';
import { useSelector } from 'react-redux';
import {CompassOutlined,BulbOutlined,MessageOutlined,CodeOutlined,SendOutlined,UserOutlined} from "@ant-design/icons"
import { VscAccount } from "react-icons/vsc";
import {useNavigate} from "react-router-dom"
import { createChatApi } from '../../../utils/client/api';
function Home() {

    const stateAuth = useSelector(state=>state.UserReducer)
    const [show,setShow] = useState(true);
    const [form] = Form.useForm();
    const navigate = useNavigate()
    const handleChatText = async(text)=>{
        const e ={
            message:text
        }
        setShow(false)
        const res = await createChatApi(e);
        
        if(res.code==200){
            navigate(`chat/${res.data._id}`)
        }
        
    }

    const handleSend = async(e)=>{

        form.resetFields()
        setShow(false)
        const res = await createChatApi(e);
        
        if(res.code==200){
            navigate(`chat/${res.data._id}`)
        }
        
        

    }

    useEffect(()=>{},[show])


    return (
        <>
            {show ? (
                <>
                    <div className='bg-white min-h-full pb-3 relative'>
                        <div className='flex items-center justify-between  p-[20px] text-[#585858]'>
                            <p className='text-[22px]'>Niko-Bot</p>
                            <Image 
                                src='https://smilemedia.vn/wp-content/uploads/2022/09/cach-chup-hinh-the-dep.jpeg' 
                                width={45} // Adjust the width and height here
                                height={45} 
                                // Disable image preview on click if not needed
                                className="rounded-[50%]"
                            />
                        </div>

                        <div className='m-auto max-w-[900px]'>
                            <div className='my-[50px] mx-[0px] text-[50px] font-medium p-[20px] text-[#c4c7c5] '>
                                <p className='uppercase' style={{
                            background: 'linear-gradient(16deg, #4b90ff, #ff5546)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent'
                                }} >Hello, {stateAuth.userInfo.fullName}</p>
                                <p>How can I help you today?</p>
                            </div> 
                    

                            <div className='' 
                                style={{
                                    display: 'grid',
                                    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                                    gap: '16px',
                                    padding:"20px"
                                }}
                            >
                                <div onClick={()=>{handleChatText("Cau hoi so 1")}} className='h-[200px] p-[15px] bg-slate-200 relative rounded-lg cursor-pointer hover:bg-slate-300 transition duration-300'>
                                    <p className='text-[15px] text-[#585858]'>Cau hoi so 1</p>
                                    <CompassOutlined className='w-[35px] text-[20px] p-[5px] absolute rounded-3xl bottom-[10px] right-[10px] '/>
                                </div>
                                <div onClick={()=>{handleChatText("Cau hoi so 1")}} className='h-[200px] p-[15px] bg-slate-200 relative rounded-lg cursor-pointer text-[20px] hover:bg-slate-300 transition duration-300'>
                                    <p className='text-[15px] text-[#585858]'>Cau hoi so 1</p>
                                    <BulbOutlined className='w-[35px] p-[5px] absolute rounded-3xl bottom-[10px] right-[10px] text-[20px]'/>
                                </div>
                                <div onClick={()=>{handleChatText("Cau hoi so 1")}} className='h-[200px] p-[15px] bg-slate-200 relative rounded-lg cursor-pointer hover:bg-slate-300 transition duration-300'>
                                    <p className='text-[15px] text-[#585858]'>Cau hoi so 1</p>
                                    <MessageOutlined className='w-[35px] p-[5px] absolute rounded-3xl bottom-[10px] right-[10px] text-[20px]'/>
                                </div>
                                <div onClick={()=>{handleChatText("Cau hoi so 1")}} className='h-[200px] p-[15px] bg-slate-200 relative rounded-lg cursor-pointer hover:bg-slate-300 transition duration-300'>
                                    <p className='text-[15px] text-[#585858]'>Cau hoi so 1</p>
                                    <CodeOutlined className='w-[35px] p-[5px] absolute rounded-3xl bottom-[10px] right-[10px] text-[20px]'/>
                                </div>
                            </div>


            

                            <div className='absolute bottom-0 w-full max-w-[900px] py-0 px-[20px] m-auto'>
                                
                                    {/* <Form className='flex center justify-between gap-[0px] bg-[#f0f4f9] rounded-[50px] py-[10px] px-[20px] '>
                                        <Form.Item name="search" className='flex-1 mb-0 h-full'>
                                            <Input className='h-full bg-transparent border-none outline-none p-[8px] text-[16px] hover:bg-transparent focus:bg-transparent focus:border-none focus:outline-none' placeholder='Enter a prompt here'/>

                                            
                                        </Form.Item>
                                        <Form.Item className='flex items-center'>
                                            <button>
                                    
                                                <SendOutlined className='text-2xl cursor-pointer'/>
                                            </button>
                                        </Form.Item>
                                    </Form> */}

                                <Form form={form} onFinish={handleSend} className='flex items-center justify-between gap-[5px] bg-[#f0f4f9] rounded-[50px] py-[10px] mb-4 px-[20px]'>
                                    <Form.Item name="message" className='flex-1 mb-0 h-full '>
                                        <Input className='h-full bg-transparent border-none outline-none focus:outline-none focus:border-none focus:bg-transparent  hover:outline-none hover:border-none hover:bg-transparent' placeholder='Enter a prompt here'/>
                                    </Form.Item>
                                    <Form.Item className='flex items-center mb-0'>
                                        <button htmltype="submit">
                                            <SendOutlined className='text-2xl cursor-pointer'/>
                                        </button>
                                    </Form.Item>
                                </Form>

                                <p className='text-[13px] my-[15px] mx-auto text-center font-light'>
                                Niko-Bot can make mistakes. Please verify important information.
                                </p>
                            </div>

                        </div>
                    </div>
                
                </>

            ):(
                <>
                    <div className='bg-white min-h-full pb-3 relative'>
                        <div className='flex items-center justify-between  p-[20px] text-[#585858]'>
                            <p className='text-[22px]'>Niko-Bot</p>
                            <Image 
                                src='https://smilemedia.vn/wp-content/uploads/2022/09/cach-chup-hinh-the-dep.jpeg' 
                                width={45} // Adjust the width and height here
                                height={45} 
                                // Disable image preview on click if not needed
                                className="rounded-[50%]"
                            />
                        </div>

                        <div className='m-auto max-w-[900px]'>
                            
                    

                            


         
                            <div className='py-0 px-[5%] max-h-[70vh] overflow-y-scroll '> 
                                
                                <div className='my-[50px] mx-0 flex items-center gap-[10px] p-2 rounded-md'>
                                    <Skeleton active/>
                                </div>
                            </div>
                            <div className='absolute bottom-0 w-full max-w-[900px] py-0 px-[20px] m-auto'>

                                <Form form={form} onFinish={handleSend} className='flex items-center justify-between gap-[5px] bg-[#f0f4f9] rounded-[50px] py-[10px] mb-4 px-[20px]'>
                                    <Form.Item name="message" className='flex-1 mb-0 h-full '>
                                        <Input className='h-full bg-transparent border-none outline-none focus:outline-none focus:border-none focus:bg-transparent  hover:outline-none hover:border-none hover:bg-transparent' placeholder='Enter a prompt here'/>
                                    </Form.Item>
                                    <Form.Item className='flex items-center mb-0'>
                                        <button htmltype="submit">
                                            <SendOutlined className='text-2xl cursor-pointer'/>
                                        </button>
                                    </Form.Item>
                                </Form>

                                <p className='text-[13px] my-[15px] mx-auto text-center font-light'>
                                Niko-Bot can make mistakes. Please verify important information.
                                </p>
                            </div>

                        </div>
                    </div>
                
                
                
                </>
            )}
        
        
        </>
        
    );
}

export default Home;