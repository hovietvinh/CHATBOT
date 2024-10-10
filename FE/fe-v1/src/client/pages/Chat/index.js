import { Form, Image, Input, Skeleton } from 'antd';
import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import {CompassOutlined,BulbOutlined,MessageOutlined,CodeOutlined,SendOutlined,UserOutlined} from "@ant-design/icons"
import { VscAccount } from "react-icons/vsc";
import {useNavigate, useParams} from "react-router-dom"
import { getChatApi, sendMessApi } from '../../../utils/client/api';
function Home() {
    
    const stateAuth = useSelector(state=>state.UserReducer)
    const [show,setShow] = useState(false);
    const [form] = Form.useForm();
    const navigate = useNavigate()
    const { id } = useParams();
    const [data,setData] = useState({})
    const [loading,setLoading] = useState(false); 
    const [reload,setReload] = useState(false); 

    const getChat = async()=>{
        const res = await getChatApi(id);
        if(res.code==200){
            setData(res.data);
        }
        setShow(true)
    }
    const handleSend =async (e)=>{
        setLoading(true);
        form.resetFields()
        e._id = id;
        const res = await sendMessApi(e);
        console.log(res);
        if(res.code==200){
            setLoading(false);
            setReload(!reload)         
        }
        

    }
    useEffect(()=>{
        getChat()
    },[id,reload])
    const endOfChatRef = useRef(null);
    useEffect(() => {
        // Scroll to the bottom of the chat or the skeleton loader
        if (endOfChatRef.current) {
            endOfChatRef.current.scrollIntoView({ behavior: 'smooth' });
        }
    }, [data.chatContent, loading]); 
    return (
        <>
            {show && (
                <>
                    <div className='bg-white min-h-full pb-3 relative'>
                        <div className='flex items-center justify-between  p-[20px] text-[#585858]'>
                            <p className='text-[22px]'>Niko-Bot</p>
                            <Image 
                                src = {stateAuth.userInfo.avatar? stateAuth.userInfo.avatar: "https://png.pngtree.com/png-vector/20220527/ourlarge/pngtree-unknown-person-icon-avatar-question-png-image_4760937.png"}

                                width={45} // Adjust the width and height here
                                height={45} 
                                // Disable image preview on click if not needed
                                className="rounded-[50%]"
                            />
                        </div>

                        <div className='m-auto max-w-[900px]'>
                            
                    

                            


         
                            <div className='py-0 px-[5%] max-h-[70vh] overflow-y-scroll '> 
                                
                                {data && data.chatContent.length>0 &&(
                                    <>
                                        {data.chatContent.map(item=>(
                                            <>
                                                <div className='my-[50px] mx-0 flex items-center gap-[10px] p-2 bg-slate-200 rounded-md'>
                                                    <UserOutlined className='rounded-[50%] text-base '/>
                                                    <p>{item.chatUser}</p>
                                                </div>

                                                <div className='flex items-start gap-[10px] '>
                                                    <VscAccount size={30}/>
                                                    <div dangerouslySetInnerHTML={{ __html: item.chatMachine }} />
                                                </div>          
                                            </>
                                        ))}
                                         <div ref={endOfChatRef} />
                                    </>
                                )}
                                {loading && (
                                                    <>
                                                    <Skeleton className='my-[50px]' active/>
                                         <div ref={endOfChatRef} />
                                                    </>
                                                )}
                            </div>

                    {/* end chat  */}




                                
                            {/* </div> */}

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

            )}
        
        
        </>
        
    );
}

export default Home;