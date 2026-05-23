# \_DART Docker Build Note

以下內容僅限在 DART 的 server 上 build 的過程筆記，若非 DART 成員請不要按照以下建議來 build container

1. 首先 git clone 到你的專案資料夾，並 cd 到此 repo 中
2. 執行以下指令建立 docker image，"sparse_ssl_image"可以替換成你喜歡的 image 名稱
   ```powershell
   docker build -t sparse_ssl_image .
   ```
   註：image 名字必須是小寫
3. 執行以下指令開始執行容器

   ```powershell
   docker run --gpus 'device=0' -d -it --rm -v /home/sharpaste/repo/LatteArt_Judge/main:/app/main -v /home/sharpaste/repo/LatteArt_Judge/runs:/app/runs --name sharpaste_SparseSSL_gpu_0 sparse_ssl_image
   ```

   ```powershell
   docker run --gpus 'device=0' -d -it --rm -v C:\Users\郭家榕\Documents\code\LatteArt_Judge\main:/app/main -v C:\Users\郭家榕\Documents\code\LatteArt_Judge\runs:/app/runs --name sharpaste_SparseSSL_gpu_0 sparse_ssl_image
   ```

   `--rm`
   會在容器停止後自動刪除容器，在開發時期較為方便，穩定佈署時可以刪除此後綴

   `-it`
   -i 或 --interactive (互動式): 保持標準輸入 (STDIN) 開啟，即使沒有連接到終端機。 這允許你與容器進行互動，例如向容器輸入命令。
   -t 或 --tty (偽終端機): 分配一個偽終端機 (pseudo-TTY)，並連接到容器的標準輸入、標準輸出和標準錯誤輸出。 這會模擬一個終端機環境，讓你可以像在一般終端機中一樣與容器互動，例如看到命令提示字元、使用方向鍵、Tab 鍵補全等等。

   `-v`
   將伺服器主機上的目錄或檔案，掛載到容器內部的指定目錄。 Volume Mount 提供了在容器和伺服器主機之間共享資料的機制。 -v /host/path:/container/path 的語法表示將伺服器上的 /host/path 掛載到容器內的 /container/path。

   /home/your_username/my_project: 伺服器主機上的目錄路徑 (Source Path)。 這是你之前 git clone 程式碼的目錄。

   :/app: 容器內部的目錄路徑 (Destination Path)。 這與你在 Dockerfile 中設定的 WORKDIR /app 相呼應。

   註：`sharpaste/repo/LatteArt_Judge`是我的專案路徑，請替換成你自己所需要的配置
   如下：

   ```powershell
   docker run --gpus all -d -it --rm -v C:/Users/a3525/Documents/program/testing/Python/LatteArt_Judge/main:/app/main -v C:/Users/a3525/Documents/program/testing/Python/LatteArt_Judge/runs:/app/runs --name sharpaste_SparseSSL_gpu_0 sparse_ssl_image
   ```

4. 查看容器日誌

   ```powershell
   docker logs sharpaste_SparseSSL_gpu_0
   ```

5. 進入容器內部
   ```powershell
   docker exec -it sharpaste_SparseSSL_gpu_0 /bin/bash
   ```
6. 關閉容器

   ```powershell
   docker stop sharpaste_SparseSSL_gpu_0
   ```

7. 離開容器

   ```powershell
   exit
   ```

8. 記得刪除容器

   ```powershell
   docker rm sharpaste_SparseSSL_gpu_0
   ```

9. 監控 GPU 用量的指令，每秒更新一次 (僅限 Nvidia 顯卡)
   ```powershell
   nvidia-smi -l 1
   ```
