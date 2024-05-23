from transformers import AutoModel, AutoTokenizer

model_name = 'Bllossom/llama-3-Korean-Bllossom-70B'
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# pip3 install torch torchvision torchaudio 

# PS C:\Users\kevin\Documents\GitHub\Langchain_Embedding> python .\modeldownload.py
# Traceback (most recent call last):
#   File "C:\Users\kevin\Documents\GitHub\Langchain_Embedding\modeldownload.py", line 1, in <module>
#     from transformers import AutoModel, AutoTokenizer
#   File "C:\Python\Python310\lib\site-packages\transformers\__init__.py", line 26, in <module>
#     from . import dependency_versions_check
#   File "C:\Python\Python310\lib\site-packages\transformers\dependency_versions_check.py", line 16, in <module>
#     from .utils.versions import require_version, require_version_core
#   File "C:\Python\Python310\lib\site-packages\transformers\utils\__init__.py", line 33, in <module>
#     from .generic import (
#   File "C:\Python\Python310\lib\site-packages\transformers\utils\generic.py", line 465, in <module>
#     import torch.utils._pytree as _torch_pytree
#   File "C:\Python\Python310\lib\site-packages\torch\__init__.py", line 141, in <module>
#     raise err
# OSError: [WinError 126] 지정된 모듈을 찾을 수 없습니다. Error loading "C:\Python\Python310\lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies.
# https://discuss.pytorch.org/t/failed-to-import-pytorch-fbgemm-dll-or-one-of-its-dependencies-is-missing/201969/2
# https://aka.ms/vs/17/release/vc_redist.x64.exe
