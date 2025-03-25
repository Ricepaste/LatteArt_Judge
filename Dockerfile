# 使用官方 PyTorch 映像檔作為基礎映像檔
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# 設定工作目錄在容器內
WORKDIR /app

# 將目前目錄下的所有檔案複製到容器的 /app 目錄
COPY . /app

# 安裝git
RUN apt-get update && apt-get install -y git

# 安裝 GradCache 套件 (分開 git clone 和 git checkout)
RUN git clone https://github.com/luyug/GradCache.git /app/third_party_libs/GradCache

RUN cd /app/third_party_libs/GradCache && git checkout 906f03835fbc183132a9db32612a9e8f180ca3b4

RUN pip install --no-cache-dir -e /app/third_party_libs/GradCache

# 安裝 requirements.txt 中的其他 Python 套件
RUN pip install --no-cache-dir -r requirements.txt

# 安裝 rigl-torch所需套件
RUN pip install --no-cache-dir -r /app/third_party_libs/rigl-torch/requirements.txt

# 安裝 rigl-torch
RUN pip install --no-cache-dir -e /app/third_party_libs/rigl-torch

# 設定容器啟動時執行的命令 (例如，運行你的 PyTorch 程式)
# CMD ["python", "your_script.py"]  # 將 "your_script.py" 替換成你的主要程式碼檔案名稱