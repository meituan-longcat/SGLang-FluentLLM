## 概述
本样例基于DispatchCombineV2Custom算子工程，介绍了单算子工程、单算子调用、第三方框架调用。
## 目录结构介绍
```
├── DispatchCombineV2Custom    // DispatchCombineV2Custom自定义算子工程 
└── DispatchCombineV2CustomSample    // DispatchCombineV2Custom自定义算子测试工程
    └── PytorchInvocation            // pytorch调用直调测试代码
```
## 算子工程介绍
其中，算子工程目录DispatchCombineV2Custom包含算子实现的模板文件、编译脚本等，如下所示:
```
├── DispatchCombineV2Custom   //DispatchCombineV2Custom自定义算子工程
│   ├── cmake
│   ├── op_host             // host侧实现文件
│   ├── op_kernel           // kernel侧实现文件
│   ├── scripts             // 自定义算子工程打包相关脚本所在目录
│   ├── build.sh            // 编译入口脚本
│   ├── CMakeLists.txt      // 算子工程的CMakeLists.txt
│   └── CMakePresets.json   // 编译配置项
```
## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程生成算子安装包；
- 安装自定义算子到算子库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 编译算子工程<a name="operatorcompile"></a>
  编译自定义算子工程，构建生成自定义算子包。
  - 完成环境准备。
  #### 安装开发套件包  
  1. 获取CANN软件包和communitysdk包（与华为生态工程师获取）。  
  2. 安装CANN软件包，安装完成后，CANN开发套件的相关组件默认存储在“/usr/local/Ascend/ascend-toolkit/latest”路径下。  
    x86_64  
    ./Ascend-cann-toolkit_{software version}_linux-x86_64.run --install   
    aarch64  
    ./Ascend-cann-toolkit_{software version}_linux-aarch64.run --install

  3. 安装communitysdk包（临时）,路径与上线的安装目录一致，后缀加opensdk  
      x86_64  
      ./Ascend-cann-communitysdk_{software version}_linux-x86_64.run --noexec --extract=/usr/local/Ascend/ascend-toolkit/latest/opensdk  
      aarch64   
      ./Ascend-cann-communitysdk_{software version}_linux-aarch64.run --noexec --extract=/usr/local/Ascend/ascend-toolkit/latest/opensdk
    
  4. （可选）使用自定义路径安装，其中{install_path}为指定的安装路径，安装完成后，CANN开发套件的相关组件存储在“${install_path}/ascend-toolkit/latest”路径下。  
      x86_64  
      ./Ascend-cann-toolkit_{software version}_linux-x86_64.run --install --install-path={install_path}  
      aarch64  
      ./Ascend-cann-toolkit_{software version}_linux-aarch64.run --install --install-path={install_path}  

  #### 算子编译与安装
  - 设置CANN运行环境变量（路径参考第一步**安装开发套件包**中安装的路径）  
  source /usr/local/Ascend/ascend-toolkit/set_env.sh  
  如果指定了安装路径，则执行source <install_path>/ascend-toolkit/set_env.sh  

  - 指定安装目录（软件包安装路径的latest目录），编译自定义算子包  
  export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest

  - 执行如下命令，切换到算子工程DispatchCombineV2Custom目录。
    ```bash
      cd $HOME/DispatchCombineV2Custom
    ```

  - 修改CMakePresets.json中ASCEND_CANN_PACKAGE_PATH为CANN软件包安装后的实际路径。
      ```json
      {
        ……
          "configurePresets": [
            {
              "ASCEND_CANN_PACKAGE_PATH": {
              "type": "PATH",
              "value": "/usr/local/Ascend/ascend-toolkit/latest"   //请替换为CANN软件包安装后的实际路径。eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest
              },
          ……
            }
          ]
      }
      ```
  - 在算子工程DispatchCombineV2Custom录下执行如下命令，进行算子工程编译。

    ```bash
    chmod -R 777 cmake/util/*
    ./build.sh
    ```
  - 编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包
  ```
  （x86_64）  
  ./build_out/custom_opp_ubuntu_x86_64.run
  （aarch64）  
  ./build_out/custom_opp_euleros_aarch64.run
  ```

  备注：如果要使用dump调试功能，需要移除op_host内和CMakeLists.txt内的Atlas 训练系列产品、Atlas 200/500 A2 推理产品的配置。

### 2. 部署算子包

执行如下命令，在自定义算子安装包所在路径下，安装自定义算子包。
  ```bash
  cd build_out
  ./custom_opp_<target os>_<target architecture>.run
  ```
命令执行成功后，自定义算子包中的相关文件将部署至当前环境的OPP算子库的vendors/customize目录中。

### 3. 配置环境变量

  这里的\$HOME需要替换为CANN包的安装路径。
  ```bash
  export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
  ```
### 4. 调用执行算子工程
- [使用pytorch调用的方式调用DispatchCombineV2Custom算子工程](../DispatchCombineV2CustomSample/PytorchInvocation/README.md)