import React, { useEffect, useState } from "react";
import RegisterForm from "./RegisterForm";
import {
  Button,
  Input,
  Dropdown,
  Menu,
  Space,
  Modal,
  Select,
  List,
  Skeleton,
} from "antd";
import {
  DownOutlined,
  ClockCircleOutlined,
  DesktopOutlined,
  DownloadOutlined,
  DashboardOutlined,
  PictureOutlined,
  InfoCircleOutlined,
  MessageOutlined,
  UserOutlined,
  SoundOutlined,
  //   PlugOutlined,
  DeleteOutlined,
  CloseOutlined,
  AppstoreOutlined,
} from "@ant-design/icons";
import axios from "axios";

const items = [
  {
    icon: <DashboardOutlined />,
    label: "모델 추가",
  },
  {
    icon: <ClockCircleOutlined />,
    label: "모델 수정",
  },
  {
    icon: <UserOutlined />,
    label: "Accounts",
  },
  {
    icon: <InfoCircleOutlined />,
    label: "About",
  },
];

const ModalLayout = ({ open, setOpen, models, fetchModals }) => {
  const [selected, setSelected] = useState(0);
  const [selectedId, setSelectedId] = useState();
  const [files, setFiles] = useState([]);

  const handleOk = () => {
    setOpen(false);
  };
  const handleCancel = () => {
    setOpen(false);
  };

  const onClick = (e) => {
    console.log("click ", e);
  };

  const handleChange = (id) => {
    setSelectedId(id);
    fetchSavedModel(id);
    // console.log(e);
  };

  const fetchSavedModel = async (modelId) => {
    const response = await axios.get(`http://localhost:5001/model/${modelId}`);
    const data = response.data;
    setFiles(data.files);
    // setSelectedName(data.label);
  };

  const deleteFilesHandler = async (modelId, fileName) => {
    const response = await axios.delete(
      `http://localhost:5001/model/file/${fileName}/id/${modelId}`
    );
    const data = response.data;
    setFiles(data.files);
    // mongodb에서만 지우고
    fetchSavedModel(modelId);
  };

  useEffect(() => {
    setFiles([]);
  }, [selected]);

  return (
    open && (
      <div className="fixed inset-0 flex items-center justify-center z-50 ">
        <div className="absolute inset-0 bg-black opacity-50"></div>
        <div className="relative bg-[white] rounded-lg shadow-lg p-6 max-w-3xl w-full animate-fadeIn">
          <header className="flex justify-between items-center pb-5 border-b-[1px]">
            <h1 className="text-xl font-semibold">Settings</h1>
            <Button
              type="text"
              icon={<CloseOutlined />}
              onClick={handleCancel}
              className=""
            />
          </header>
          {/* side */}
          <div className="flex min-h-[28rem]">
            <aside className="flex flex-col w-48 pr-4 py-4 space-y-2 border-r-[1px]">
              {items.map((val, idx) => {
                return (
                  <button
                    type="text"
                    className="self-start w-full flex hover:bg-[#e7e7e7] p-2 rounded-md"
                    style={{ backgroundColor: selected === idx && "#e7e7e7" }}
                    onClick={() => {
                      setSelected(idx);
                    }}
                  >
                    <div className="mr-4">{val.icon}</div>
                    <div>{val.label}</div>
                  </button>
                );
              })}
            </aside>
            {/* 메인 modal */}
            <section className="flex flex-col flex-grow p-4 space-y-4 overflow-auto">
              {selected === 0 ? (
                <RegisterForm setOpen={setOpen} fetchModals={fetchModals} />
              ) : selected === 1 ? (
                <div className="space-y-4">
                  <div className="space-y-2">
                    <h3 className="text-lg font-semibold">
                      모델 학습용 파일 수정
                    </h3>
                  <div className="flex">
                    <Select
                      placeholder={"선택해주세요"}
                      //   defaultValue={models.filter((val) => val.files)[0].label}
                      className="w-full flex items-center justify-between"
                      onChange={handleChange}
                      options={models
                        .filter((val) => val.files)
                        .map((val) => {
                          return {
                            value: val.id,
                            label: val.label,
                          };
                        })}
                    />
                    {/* <Button
                      type="text"
                      icon={<DeleteOutlined />}
                      //   className="text-white"
                    /> */}
                    </div>
                    {files && (
                      <List
                        itemLayout="horizontal"
                        dataSource={files || []}
                        renderItem={(item) => (
                          <List.Item>
                            <Skeleton
                              avatar
                              title={false}
                              loading={item.loading}
                              active
                            >
                              <List.Item.Meta
                                // avatar={<Avatar src={item.picture.large} />}
                                title={
                                  <div className="flex w-full justify-between">
                                    <div>{item}</div>
                                    {files.length > 1 && (
                                      <Button
                                        onClick={() => {
                                          deleteFilesHandler(selectedId, item);
                                        }}
                                      >
                                        삭제
                                      </Button>
                                    )}
                                  </div>
                                  // </Checkbox>
                                }
                              />
                            </Skeleton>
                          </List.Item>
                        )}
                      />
                    )}

                    {/* <Select className="flex items-center justify-between p-2 rounded-md">
                      <span>Select a model</span>
                      <div className="flex space-x-2">
                        <Button
                          type="text"
                          icon={<DownOutlined />}
                          className="text-white"
                        />
                        <Button
                          type="text"
                          icon={<DeleteOutlined />}
                          className="text-white"
                        />
                      </div>
                    </Select> */}
                  </div>
                </div>
              ) : (
                "작업중"
              )}
            </section>
          </div>
        </div>
      </div>
    )
  );
};
export default ModalLayout;
