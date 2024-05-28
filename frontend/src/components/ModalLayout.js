import React, { useState } from "react";
import RegisterForm from "./RegisterForm";
import { Button, Input, Dropdown, Menu, Space, Modal } from "antd";
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

const items = [
  {
    icon: <DashboardOutlined />,
    label: "모델",
  },
  {
    icon: <ClockCircleOutlined />,
    label: "작업 중",
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

const ModalLayout = ({ open, setOpen, fetchModals }) => {
  const [selected, setSelected] = useState(0);

  const handleOk = () => {
    setOpen(false);
  };
  const handleCancel = () => {
    setOpen(false);
  };

  const onClick = (e) => {
    console.log("click ", e);
  };

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
                    <h2 className="text-lg font-semibold">
                      Manage Ollama Models
                    </h2>
                    <div className="flex items-center justify-between bg-[#e7e7e7] p-2 rounded-md">
                      <span>http://localhost:11434</span>
                      <div className="flex space-x-2">
                        <Button
                          type="text"
                          icon={<DownOutlined />}
                          className="text-white"
                        />
                        <Button
                          type="text"
                          icon={<ClockCircleOutlined />}
                          className="text-white"
                        />
                      </div>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-lg font-semibold">
                      Pull a model from Ollama.com
                    </h3>
                    <div className="flex items-center bg-[#e7e7e7] p-2 rounded-md">
                      <Input
                        className="bg-transparent text-white"
                        placeholder="Enter model tag (e.g. mistral:7b)"
                      />
                      <Button
                        type="text"
                        icon={<DownloadOutlined />}
                        className="text-white"
                      />
                    </div>
                    <p className="text-sm">
                      To access the available model names for downloading,
                      <a className="text-blue-500" href="#">
                        {" "}
                        click here
                      </a>
                    </p>
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-lg font-semibold">Delete a model</h3>
                    <div className="flex items-center justify-between bg-[#e7e7e7] p-2 rounded-md">
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
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">Experimental</h3>
                    <Button type="text" className="text-white">
                      Show
                    </Button>
                  </div>
                  <div className="flex items-center justify-between">
                    <h3 className="text-lg font-semibold">
                      Manage LiteLLM Models
                    </h3>
                    <Button type="text" className="text-white">
                      Show
                    </Button>
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
