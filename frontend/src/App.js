import React, { useState, useEffect, useRef } from "react";
import axios from "axios";

const MyComponent = () => {
  const URL = process.env.REACT_APP_URL;
  const [prompt, setPrompt] = useState(
    "write a short koan story about seeing beyond"
  );
  const [output, setOutput] = useState("");
  const aborterRef = useRef(new AbortController());

  const run = async () => {
    if (aborterRef.current) {
      aborterRef.current.abort(); // Cancel the previous request
    }
    setOutput("");
    aborterRef.current = new AbortController();

    try {
      const response = await fetch(`${URL}/chain`, {
        signal: aborterRef.current.signal,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const decoded = decoder.decode(value, { stream: true });
        setOutput((prev) => prev + decoded);
      }
      // const response = await axios.post(`${URL}/chat`, {
      //   prompt,
      // });
      // const generatedText = response.data.answer;
      // setOutput(generatedText);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error(err);
      }
    }
  };
  const handleSubmit = () => {
    run();
  };

  return (
    <div className="flex h-screen bg-[#121212]">
      <div className="flex flex-col w-64 bg-[#1E1E1E] p-4 space-y-4">
        <button className="bg-[#333333] text-white">New Chat</button>
        <div className="flex flex-col space-y-2">
          <a className="text-gray-300" href="#">
            Model files
          </a>
          <a className="text-gray-300" href="#">
            Prompts
          </a>
          <a className="text-gray-300" href="#">
            Documents
          </a>
        </div>
      </div>
      <div className="flex flex-col flex-1 p-4">
        <div className="flex items-center justify-between mb-4">
          <button className="text-gray-400" variant="ghost">
            Set as default
          </button>
        </div>
        <div className="flex-1 overflow-hidden">
          <div className="h-full">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                {/* <Badge variant="secondary">Llava:latest (4.4GB)</Badge> */}
                {/* <ChevronDownIcon className="text-gray-400" /> */}
              </div>
              {/* <div className="text-white">안녕 챗봇</div> */}
              <div className="text-white">{output}</div>

              {/* <div
                className="flex items-center space-x-2 "
                style={{ width: "-webkit-fill-available" }}
              > */}
              <div
                className="flex items-center justify-between bg-[#333333] p-4 rounded-lg fixed bottom-2 mr-4"
                style={{ width: "-webkit-fill-available" }}
              >
                <div className="flex items-center w-full">
                  <PlusIcon className="mr-2 text-white w-fit" />
                  <div className="">
                    <input
                    className="w-full min-w-[50rem]"
                      placeholder="text"
                      onChange={(e) => setPrompt(e.target.value)}
                    />
                  </div>
                  <button
                    onClick={() => handleSubmit()}
                    className="border ml-4 p-1 bg-white"
                  >
                    전송
                  </button>
                  <div className="w-fit flex">
                    <MicIcon className="text-white mx-4" />
                    <SettingsIcon className="text-white" />
                  </div>
                </div>
              </div>
              {/* </div> */}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

function MicIcon(props) {
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
      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
      <line x1="12" x2="12" y1="19" y2="22" />
    </svg>
  );
}

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

function SettingsIcon(props) {
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
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function ChevronDownIcon(props) {
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
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function PlaneIcon(props) {
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
      <path d="M17.8 19.2 16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
    </svg>
  );
}

export default MyComponent;
