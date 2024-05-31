import React, { useState } from "react";
import { Button, Input, Modal } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import { message, Upload } from "antd";
import axios from "axios";
const { Dragger } = Upload;

const RegisterForm = ({ setOpen, fetchModals }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [selectedName, setSelectedName] = useState("");

  const onChange = (e) => {
    setSelectedName(e.target.value);
  };

  const normFile = (file) => {
    // 50 메가바이트 제한 × 1024 킬로바이트/메가바이트 × 1024 바이트/킬로바이트 = 52,428,800 바이트
    // 토탈 200메가 제한
    const size = 52428800;
    const totalSize = size * 4;
    // const size = 1000;
    const sizes = file.fileList.map((val) => {
      return val.size >= size ? 1 : 0;
    });
    const sum = file.fileList.reduce((acc, cur, idx) => {
      const totalSize = acc + cur.size; // Use a different variable name for the accumulator
      return totalSize;
    }, 0);
    setSelectedFiles(file.fileList);
    // const sum = 1
  };

  const fileSubmitHandler = async () => {
    const data = new FormData();
    for (const file of selectedFiles) {
      const blob = new Blob([file.originFileObj], { type: file.type });
      // Blob에 원래 파일 이름을 설정하는 부분은 불필요
      // 대신 FormData에 파일을 추가할 때 파일 이름을 명시적으로 설정
      data.append("files", blob, file.name);
    }

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_URL}/upload/${selectedName}`,
        data,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setSelectedName("");
      setSelectedFiles([]);
      return true;
    } catch (error) {
      return false;
    }
  };

  return (
    <>
      <div className="mb-1">
        <p className="mb-2">모델 이름 입력</p>
        <Input
          showCount
          maxLength={20}
          onChange={onChange}
          value={selectedName}
        />
        {/* <input className="flex items-center justify-between bg-[#333333] p-2 rounded-md" value={selectedName} onChange={onChange}/> */}
      </div>
      <div className="flex justify-between">
        <p className="mb-2">모델 학습용 파일 업로드</p>
        <Button
          key="submit"
          type="primary"
          // loading={loading}
          disabled={selectedFiles.length === 0}
          onClick={async () => {
            const result = await fileSubmitHandler();
            console.log('result', result, process.env.REACT_APP_URL)
            if (result) {
              await fetchModals();
              setOpen(false);
            } else {
              alert("오류 발생");
            }
          }}
        >
          추가
        </Button>
      </div>
      <Dragger
        multiple
        customRequest={({ file, onSuccess, onError }) => {
          Promise.resolve().then(() => onSuccess());
        }}
        style={{ maxWidth: "652px", maxHeight: "5rem !important" }}
        beforeUpload={false}
        onChange={(e) => normFile(e)}
        fileList={selectedFiles}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        {/* <p className="ant-upload-hint text-white">
          여러 파일을 한 번에 드랍하여 업로드해 주세요.
        </p> */}
      </Dragger>
      <div className="flex justify-end"></div>
    </>
  );
};
export default RegisterForm;
