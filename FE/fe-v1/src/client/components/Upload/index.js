import React, { useState, useEffect } from 'react';
import { LoadingOutlined, PlusOutlined } from '@ant-design/icons';
import { message, Upload } from 'antd';

const beforeUpload = (file) => {
  const isJpgOrPng = file.type === 'image/jpeg' || file.type === 'image/png';
  if (!isJpgOrPng) {
    message.error('You can only upload JPG/PNG file!');
    return Upload.LIST_IGNORE;
  }
  const isLt2M = file.size / 1024 / 1024 < 2;
  if (!isLt2M) {
    message.error('Image must be smaller than 2MB!');
    return Upload.LIST_IGNORE;
  }
  return true;
};

const getBase64 = (file, callback) => {
  const reader = new FileReader();
  reader.addEventListener('load', () => callback(reader.result));
  reader.readAsDataURL(file);
};

const UploadImg = ({ onImageSelect, defaultImage }) => {
  const [loading, setLoading] = useState(false);
  const [imageUrl, setImageUrl] = useState(defaultImage || null);

  useEffect(() => {
    setImageUrl(defaultImage); // Update imageUrl when defaultImage changes
  }, [defaultImage,onImageSelect]);

  const handleChange = (info) => {
    if (info.file.status === 'done') {
      const file = info.file.originFileObj;
      getBase64(file, (url) => {
        setImageUrl(url);
      });
      onImageSelect(file);
    }
  };

  const customRequest = ({ file, onSuccess }) => {
    getBase64(file, (url) => {
      setImageUrl(url);
      onImageSelect(file);
      onSuccess();
    });
  };

  const uploadButton = (
    <div>
      {loading ? <LoadingOutlined /> : <PlusOutlined />}
      <div style={{ marginTop: 8 }}>Upload</div>
    </div>
  );

  return (
    <Upload
      name="avatar"
      listType="picture-card"
      className="avatar-uploader hidden"
      showUploadList={false}
      beforeUpload={beforeUpload}
      onChange={handleChange}
      customRequest={customRequest}
    >
      {imageUrl ? (
        <img
          src={imageUrl}
          alt="avatar"
          style={{ width: '100%', height: '100%' }}
        />
      ) : (
        uploadButton
      )}
    </Upload>
  );
};

export default UploadImg;
