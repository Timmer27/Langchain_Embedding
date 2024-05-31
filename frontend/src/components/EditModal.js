import React, { useEffect, useState } from "react";
import { Button, Checkbox, Input, List, Modal, Skeleton } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import { message, Upload } from "antd";
import axios from "axios";
const { Dragger } = Upload;

const EditModal = ({ modelId, open, setOpen }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [files, setFiles] = useState([]);
  const [selectedName, setSelectedName] = useState();

  const onChange = (e) => {
    setSelectedName(e.target.value);
  };

  const handleOk = () => {
    setOpen(false);
  };
  const handleCancel = () => {
    setSelectedFiles([]);
    setSelectedName("");
    setOpen(false);
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
    console.log("file", file);
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

  const fetchSavedModel = async () => {
    const response = await axios.get(`${process.env.REACT_APP_URL}/model/${modelId}`);
    const data = response.data;
    console.log(data);
    setFiles(data.files);
    setSelectedName(data.label);
  };

  useEffect(() => {
    fetchSavedModel();
  }, [modelId]);

  return (
    selectedName && (
      <>
        <Modal
          open={open}
          title="개인 챗봇 수정"
          onOk={handleOk}
          onCancel={handleCancel}
          footer={[
            // <Button
            //   key="submit"
            //   type="primary"
            //   // loading={loading}
            //   onClick={async () => {
            //     (await fileSubmitHandler())
            //       ? handleOk() && alert("성공!")
            //       : alert("오류 발생");
            //   }}
            // >
            //   수정
            // </Button>,
            <Button key="취소" type="primary" onClick={handleCancel}>
              취소
            </Button>,
          ]}
        >
          <div className="mb-5 mt-8">
            <p className="mb-2 text-gray-600">모델 이름 입력</p>
            <Input
              showCount
              maxLength={20}
              onChange={onChange}
              value={selectedName}
            />
          </div>
          <p className="mb-2 text-gray-600">첨부된 파일</p>
          <List
            itemLayout="horizontal"
            dataSource={files || []}
            renderItem={(item) => (
              <List.Item>
                <Skeleton avatar title={false} loading={item.loading} active>
                  <List.Item.Meta
                    // avatar={<Avatar src={item.picture.large} />}
                    title={
                      // <Checkbox
                      //   onChange={(e) => {
                      //     if (e.target.checked) {
                      //       const files = [...selectedFiles, item.FILE_DB_NAME];
                      //       setSelectedFiles(files);
                      //     } else {
                      //       const files = selectedFiles.filter(
                      //         (fileName) => fileName !== item.FILE_DB_NAME
                      //       );
                      //       setSelectedFiles(files);
                      //     }
                      //   }}
                      //   checked={selectedFiles.includes(item.FILE_DB_NAME)}
                      // >
                        // {item}
                        <div className="flex w-full justify-between">
                          <div>
                          {item}
                        </div>
                        <Button>삭제</Button>
                        </div>
                      // </Checkbox>
                    }
                  />
                </Skeleton>
              </List.Item>
            )}
          />
          {/* <p className="mb-2 text-gray-600">모델 학습용 파일 업로드</p> */}
          {/* <Dragger
          multiple
          customRequest={({ file, onSuccess, onError }) => {
            Promise.resolve().then(() => onSuccess());
          }}
          style={{ maxWidth: "652px" }}
          beforeUpload={false}
          onChange={(e) => normFile(e)}
          fileList={selectedFiles}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-hint">
            여러 파일을 한 번에 드랍하여 업로드해 주세요.
          </p>
        </Dragger> */}
        </Modal>
      </>
    )
  );
};
export default EditModal;
