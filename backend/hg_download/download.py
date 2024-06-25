import argparse, os, subprocess, shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from llama_model_quantizer.convert_hf_to_gguf import convert_model

def delete_folder_contents(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Iterate over all the files and subdirectories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    # Delete the file
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    # Recursively delete subdirectory and its contents
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
        # After deleting all files and subdirectories, delete the main folder itself
        try:
            shutil.rmtree(folder_path)
            print(f"{folder_path} and all its contents have been successfully deleted.")
        except Exception as e:
            print(f"Failed to delete {folder_path}. Reason: {e}")
    else:
        print(f"The folder {folder_path} does not exist.")


def download_hg_model(model_id:str="kyujinpy/KoR-Orca-Platypus-13B", local_dir:str="/"):
    snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False, revision="main")

def create_modelfile_and_pull_into_ollama(abs_path, model_path, modelName, outtype):
    outfile = os.path.join(abs_path, f"{modelName}.gguf")
    content = f'''FROM {modelName}.gguf

TEMPLATE """{{ if .System }}
<|start_header_id|>{{ .System }}<|end_header_id|>
{{ end }}
<|start_header_id|>Human:
{{ .Prompt }}<|end_header_id|>
<s>Assistant:
"""

SYSTEM """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
"""

PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|reserved_special_token"
'''
    # Create the file and write the content to it
    # os.chmod(model_path, 0o777)
    with open(model_path+".txt", "w") as file:
        file.write(content)

    if os.path.exists(model_path) == False:
        os.rename(model_path+".txt", model_path)

    convert_model(model_path=Path(model_path+"Dir"), outfile=Path(outfile), outtype=outtype)

    command = ["ollama", "create", modelName, "-f", f"{os.path.join(abs_path, 'Modelfile')}"]
    subprocess.run(command)

    # Delete the file
    os.remove(file_name)
    delete_folder_contents(model_path+"Dir")

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(abs_path, "Modelfile")    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modelId", type=str, help="hugging face model id. i.e. kyujinpy/KoR-Orca-Platypus-13B",
    )
    parser.add_argument(
        "--modelName", type=str, help="model name to save. i.e. KoR-Orca-Platypus-13B",
    )
    parser.add_argument(
        "--outtype", type=str, choices=["f32", "f16", "bf16", "q8_0", "auto"], default="f16",
        help="output format - use f32 for float32, f16 for float16, bf16 for bfloat16, q8_0 for Q8_0, auto for the highest-fidelity 16-bit float type depending on the first loaded tensor type",
    )
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    model_id, model_name, outtype = opt.modelId, opt.modelName, opt.outtype

    # print('model_id, model_name, outtype', model_id, model_name, outtype)

    # 모델파일 생성 및 pull
    # 실행 명령어 예시 -> python download.py --modelId openbmb/MiniCPM-Llama3-V-2_5 --modelName MiniCPM-Llama3 --outtype q8_0
    download_hg_model(model_id=model_id, local_dir=file_name+"Dir")
    create_modelfile_and_pull_into_ollama(abs_path=abs_path, model_path=file_name, modelName=model_name, outtype=outtype)
