import React, { useState } from 'react';
import {Button, Form, Input} from "antd"
import UploadImg from '../../components/Upload';
import { Link, useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { userRegisterApi } from '../../../utils/client/api';

function Register() {
    const navigate = useNavigate()
  const [imageFile, setImageFile] = useState(null);  // State for image file
    const handleImageSelect = (file) => {
        setImageFile(file);  // Save the image file when selected
    };
    const DeleteImg = ()=>{
        // console.log(imageFile);
        setImageFile(null)
    }
    const [form] = Form.useForm()
    const handleRegister = async(values)=>{
        console.log(values);
        let formData = new FormData();
        Object.keys(values).forEach(key => {
            formData.append(key, values[key] || '');
        });
        if (imageFile) {
            formData.set('avatar', imageFile);
        }
                const res = await userRegisterApi(formData);
        if(res.code==200){
            // toast.success(res.message)
            toast.success(res.message);

            navigate("/users/login")
        }
        else{
            toast.error(res.message);
        }
        form.resetFields()
    }
    return (
        <>
    
            <div className='mt-5'>
                <div className='bg-white w-full max-w-sm rounded overflow-hidden p-4 mx-auto'>
                    <h3>Welcome to Chat app!</h3>
                    <Form
                    layout='vertical'
                    className='mt-5'
                    form={form}
                    onFinish={handleRegister}             
                    >
                        <Form.Item name="fullName" label="Name :" className='' 
                            rules={[{ required: true, message: 'Please fill in this field!' }]}
                        >
                            <Input 
                                placeholder='enter your name'
                                className='bg-slate-100 px-2 py-1'
                                
                            />
                        </Form.Item>
                        <Form.Item name="email" label="Email :" className=''
                            rules={[{ required: true, message: 'Please fill in this field!' }, { type: 'email', message: 'Invalid email format!' },]}         
                        >
                            <Input 
                                placeholder='enter your email'
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
                        <Form.Item 
                            name="avatar" 
                            label={
                                <span className=''>Avatar :</span> // Thiết lập class cho label
                            } 
                            className=''
                        >
                            <div className='flex items-start'>
                                <UploadImg onImageSelect={handleImageSelect} className="hidden"/>
                                {imageFile && (
                                    <>
                                        <button onClick={DeleteImg} className='flex  mt-[-5px] ml-[-5px] hover:outline-primary items-center justify-center p-2 w-[20px] h-[20px] rounded-full border bg-slate-200 '>X</button>
                                    </>
                                )}
                            </div>   
                        </Form.Item>
                        <Form.Item>
                            <button className='w-full font-semibold text-white text-[16px] bg-primary px-4 py-1 transition-all duration-300 rounded-md hover:bg-secondary ' htmlType="submit">Register</button>
                        </Form.Item>
                        <p className='my-3 text-center'>Already have account ? <Link to={"/users/login"} className="transition duration-300 hover:text-primary font-medium ">Login</Link></p>
                    </Form>
                </div>
            </div>
        </>
    );
}

export default Register;