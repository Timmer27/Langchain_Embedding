import React, { useState, useEffect, useRef } from "react";
import {
  Menu,
  MenuButton,
  MenuItem,
  MenuItems,
  Transition,
} from "@headlessui/react";
import { BoltIcon, ChevronDownIcon, bolt } from "@heroicons/react/20/solid";
import axios from "axios";
import chatBot from "./assets/chatbot.png";
import "./App.css";

const MODELS = [
  {
    key: "1",
    label: "Open AI",
  },
  {
    key: "2",
    label: "GPT4 ALL",
  },
  {
    key: "3",
    label: "Ollama",
  },
  {
    key: "4",
    label: "커스텀 데이터 모델 (Chroma)",
  },
];

const chats = [
  {
    type: "user",
    text: ["안녕"],
  },
  {
    type: "bot",
    text: ["안녕하세요. 점심 뭐 드셨어요"],
  },
  {
    type: "user",
    text: ["아무말 해봐"],
  },
  {
    type: "bot",
    text: [
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum",
      "Forget about having to attribute an illustration with a Premium Subscription. Use any illustration either for commercial or personal use without crediting the author.",
    ],
  },
];

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

const MyComponent = () => {
  const bottomRef = useRef(null);
  const URL = process.env.REACT_APP_URL;
  const [prompt, setPrompt] = useState();
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [output, setOutput] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const aborterRef = useRef(new AbortController());

  const run = async () => {
    if (!prompt) {
      // 입력값 없을 시 아무것도 안함
      return false;
    } else if (aborterRef.current) {
      aborterRef.current.abort(); // Cancel the previous request
    }
    const userData = {
      type: "user",
      text: [prompt],
    };
    setOutput((prev) => [...prev, userData]);
    setPrompt("");
    setIsLoading(true);
    aborterRef.current = new AbortController();
    try {
      const response = await fetch(`${URL}/chain/${selectedModel.key}`, {
        signal: aborterRef.current.signal,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let words = "";

      // Initialize the botData object
      const botData = {
        type: "bot",
        text: [words],
      };
      setOutput((prev) => [...prev, botData]);
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          setIsLoading(false);
          break;
        }
        const decoded = decoder.decode(value, { stream: true });
        words += decoded;

        // Update the text data in the last index of the output array
        setOutput((prev) => {
          const updatedOutput = [...prev];
          const lastElement = updatedOutput[updatedOutput.length - 1];
          lastElement.text = [words]; // Update the text array with the new words
          return updatedOutput;
        });
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error(err);
      }
    }
  };
  const handleSubmit = () => {
    run();
  };

  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [output]);

  return (
    // layout
    <div className="flex h-screen bg-[#121212]">
      {/* sidebar */}
      <div className="flex flex-col bg-[#f9f9f9] w-64 p-4 space-y-4">
        <button className="bg-[#616161] rounded-lg py-3 text-white">
          New Chat
        </button>
        <div className="text-black flex flex-col space-y-2">
          <a className="" href="#">
            Model files
          </a>
          <a className="" href="#">
            Prompts
          </a>
          <a className="" href="#">
            Documents
          </a>
        </div>
      </div>
      {/* sidebar */}
      {/* main container */}
      <div className="bg-white flex flex-col flex-1 p-4">
        {/* model selection */}
        <section className="mx-auto mb-4 juice:gap-4 juice:md:gap-6 md:max-w-[40rem] md:min-w-[40rem] lg:max-w-[48rem] lg:min-w-[48rem] xl:max-w-[48rem] xl:min-w-[48rem] justify-end flex">
          <Menu as="div" className="relative inline-block text-left">
            <div>
              <MenuButton className="inline-flex w-full justify-center gap-x-1.5 rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 shadow-sm ring-1 ring-inset ring-gray-300 hover:bg-gray-50">
                {selectedModel.label}
                <ChevronDownIcon
                  className="-mr-1 h-5 w-5 text-gray-400"
                  aria-hidden="true"
                />
              </MenuButton>
            </div>

            <Transition
              enter="transition ease-out duration-100"
              enterFrom="transform opacity-0 scale-95"
              enterTo="transform opacity-100 scale-100"
              leave="transition ease-in duration-75"
              leaveFrom="transform opacity-100 scale-100"
              leaveTo="transform opacity-0 scale-95"
            >
              <MenuItems className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                <div className="py-1">
                  {MODELS.map((val) => {
                    return (
                      <MenuItem>
                        {({ focus }) => (
                          <a
                            onClick={() => {
                              setSelectedModel(val);
                            }}
                            href="#"
                            className={classNames(
                              focus
                                ? "bg-gray-100 text-gray-900"
                                : "text-gray-700",
                              "block px-4 py-2 text-sm"
                            )}
                          >
                            {val.label}
                          </a>
                        )}
                      </MenuItem>
                    );
                  })}
                </div>
              </MenuItems>
            </Transition>
          </Menu>
        </section>
        {/* propmt output section */}
        <section className="flex-1 overflow-hidden overflow-y-auto max-h-[83%]">
          <div className="mx-auto flex flex-col flex-1 gap-7 text-base juice:gap-4 juice:md:gap-6 md:max-w-3xl lg:max-w-[40rem] xl:max-w-[48rem] mt-5">
            {/* 프롬프트 결과값 아웃풋 */}
            {output &&
              output.map((val, index) => {
                if (val.type === "bot") {
                  return (
                    // 봇일 때
                    <div key={index} className="flex gap-7">
                      <div className="flex-shrink-0 flex flex-col relative items-end pt-2">
                        {/* 봇 이미지 */}
                        <img src={chatBot} width={33} />
                      </div>
                      <div className="w-full">
                        {val.text.map((words, i) => (
                          <p
                            key={i}
                            className="markdown prose w-full break-words dark:prose-invert light mt-3 mb-3"
                          >
                            {words}
                          </p>
                        ))}
                      </div>
                    </div>
                  );
                } else {
                  // 유저일경우
                  return (
                    <div key={index} className="w-full text-end">
                      {val.text.map((words, i) => (
                        <p
                          key={i}
                          className="text-end w-auto float-right rounded-[10px] py-2.5 px-4 bg-[rgba(128,128,128,0.18)] markdown prose break-words dark:prose-invert light"
                        >
                          {words}
                        </p>
                      ))}
                    </div>
                  );
                }
              })}
            <div ref={bottomRef} />
            {isLoading && <p className="pb-4">Loading...</p>}
          </div>
        </section>
        {/* propmt output section */}
        {/* 아래 프롬프트 입력칸 */}
        <section className="mx-auto mb-4 juice:gap-4 juice:md:gap-6 md:max-w-3xl md:min-w-3xl lg:max-w-[40rem] lg:min-w-[40rem] xl:max-w-[48rem] xl:min-w-[40rem] justify-end flex">
          <div className="flex items-center bg-[#f4f4f4] h-[3em] px-4 rounded-3xl left-1/2 transform min-w-[40em]">
            <div className="flex items-center w-full h-full justify-between">
              <div className="flex justify-between flex-1 h-full items-center">
                <svg
                  onClick={() => {
                    alert("준비 중!");
                  }}
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="size-6 mr-4 w-fit cursor-pointer"
                >
                  <path
                    fillRule="evenodd"
                    d="M18.97 3.659a2.25 2.25 0 0 0-3.182 0l-10.94 10.94a3.75 3.75 0 1 0 5.304 5.303l7.693-7.693a.75.75 0 0 1 1.06 1.06l-7.693 7.693a5.25 5.25 0 1 1-7.424-7.424l10.939-10.94a3.75 3.75 0 1 1 5.303 5.304L9.097 18.835l-.008.008-.007.007-.002.002-.003.002A2.25 2.25 0 0 1 5.91 15.66l7.81-7.81a.75.75 0 0 1 1.061 1.06l-7.81 7.81a.75.75 0 0 0 1.054 1.068L18.97 6.84a2.25 2.25 0 0 0 0-3.182Z"
                    clipRule="evenodd"
                  />
                </svg>

                <div className="w-full h-full items-center">
                  <input
                    style={{
                      outline: "none",
                    }}
                    className="w-full bg-[#f4f4f4] h-full"
                    placeholder="메세지를 입력해주세요"
                    onChange={(e) => setPrompt(e.target.value)}
                    value={prompt}
                    onKeyDown={(e) => {
                      // if (e.key === "Enter" && !isLoading) {
                        if (e.key === "Enter") {
                        handleSubmit();
                      }
                    }}
                  />
                </div>
              </div>
              <svg
                onClick={() => !isLoading && handleSubmit()}
                className="cursor-pointer"
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                width="24"
                height="24"
              >
                <path d="M4 12l1.41 1.41L11 7.83V20h2V7.83l5.59 5.58L20 12l-8-8-8 8z" />
              </svg>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
};

function PlusIcon(props) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M5 12h14" />
      <path d="M12 5v14" />
    </svg>
  );
}

export default MyComponent;
