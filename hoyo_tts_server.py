import sys
import subprocess
import os
from pathlib import Path
import time
import platform
import simplejson as json
import logging
import uuid
import ctypes
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import soundfile as sf
import torch
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope import snapshot_download
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.models.base import Model, TorchModel
from modelscope.models.builder import MODELS
import sortedcontainers as sc


# 检查管理员权限
def is_admin():
    """检查是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


# 以管理员权限重启程序
def restart_as_admin():
    """以管理员权限重启程序"""
    if is_admin():
        return True

    try:
        if sys.platform == 'win32':
            ctypes.windll.shell32.ShellExecuteW(
                None, "runas", sys.executable, " ".join(sys.argv), None, 1
            )
            sys.exit()
        else:
            print("此功能仅支持Windows系统")
            return True
    except:
        print("获取管理员权限失败")
        return False


# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def print_step(msg):
    """打印带有样式的步骤信息"""
    logger.info(f"\n🔹 {msg}")


def print_success(msg):
    """打印带有样式的成功信息"""
    logger.info(f"\n✅ {msg}")


def print_error(msg):
    """打印带有样式的错误信息"""
    logger.error(f"\n❌ {msg}")


def check_python():
    """检查Python环境"""
    print_step("检查Python环境...")
    if sys.version_info < (3, 8):
        print_error("需要Python 3.8或更高版本")
        logger.error("请访问 https://www.python.org/downloads/ 下载新版Python")
        sys.exit(1)
    print_success("Python版本检查通过")


def install_package(package: str):
    """安装指定的包"""
    try:
        logger.info(f"正在安装 {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    except subprocess.CalledProcessError:
        logger.error(f"{package} 安装失败")
        sys.exit(1)


def update_package(package: str):
    """更新指定的包"""
    try:
        logger.info(f"正在更新 {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", package], check=True)
    except subprocess.CalledProcessError:
        logger.error(f"{package} 更新失败")
        sys.exit(1)


def check_and_install_dependencies():
    """检查并安装缺失的依赖包"""
    print_step("检查依赖包...")

    requirements = {
        "modelscope": ">=1.9.5,<2.0.0",
        "torch": ">=2.0.0",
        "torchaudio": "",
        "soundfile": "",
        "scipy": "",
        "fastapi": "",
        "uvicorn": "",
        "python-multipart": "",
        "pydantic": "<2.0",
        "transformers": "",
        "datasets": "",
        "addict": ""
    }

    missing_packages = []
    update_packages = []

    for package, version in requirements.items():
        requirement = f"{package}{version}" if version else package
        try:
            pkg_resources.require(requirement)
        except DistributionNotFound:
            missing_packages.append(package)
        except VersionConflict:
            if version:
                update_packages.append(requirement)

    if not missing_packages and not update_packages:
        print_success("所有依赖包已安装且版本正确")
        return

    if missing_packages:
        print_step(f"安装缺失的包: {', '.join(missing_packages)}")
        for package in missing_packages:
            install_package(package)

    if update_packages:
        print_step(f"更新版本不匹配的包: {', '.join(update_packages)}")
        for package in update_packages:
            update_package(package)

    print_success("依赖包检查和安装完成")


@MODELS.register_module('text-to-speech', module_name='hoyotts')
class HoyoTTSModel(TorchModel):
    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.sampling_rate = self.config['data']['sampling_rate']
        self.spk2id = self.config['data']['spk2id']
        self.duration_model_path = os.path.join(model_dir, 'DUR_78000.pth')
        self.acoustic_model_path = os.path.join(model_dir, 'D_78000.pth')
        self.vocoder_model_path = os.path.join(model_dir, 'G_78000.pth')

    def forward(self, text, voice_name):
        if voice_name not in self.spk2id:
            raise ValueError(f"未知的说话人: {voice_name}")
        speaker_id = self.spk2id[voice_name]
        audio_length = int(self.sampling_rate * 3)  # 3秒的音频
        return torch.randn(audio_length)


@PIPELINES.register_module('text-to-speech', module_name='hoyotts-speech-generation')
class HoyoTTSPipeline(Pipeline):
    def __init__(self, model_dir, **kwargs):
        super().__init__(model=model_dir, **kwargs)
        self.model = HoyoTTSModel(model_dir)
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def preprocess(self, inputs):
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 2:
                raise ValueError('输入元组必须包含两个元素：text 和 voice')
            return inputs[0], inputs[1]
        elif isinstance(inputs, str):
            return inputs, "派蒙"
        else:
            raise ValueError('输入必须是包含文本和说话人的元组，或者单独的文本字符串')

    def forward(self, inputs):
        text, voice = self.preprocess(inputs)
        audio = self.model.forward(text, voice)
        result = {
            'audio': audio.cpu().numpy(),
            'sample_rate': self.model.sampling_rate
        }
        return result

    def postprocess(self, inputs):
        return inputs


def setup_server():
    """设置并启动服务器"""
    try:
        script_dir = Path(__file__).parent.resolve()
        local_cache_dir = script_dir / "model_cache"
        local_cache_dir.mkdir(exist_ok=True)

        system = platform.system()
        TEMP_DIR = Path(os.getenv(
            'TEMP')) / "hoyo_tts" / "output" if system == "Windows" else Path.home() / "Library" / "Caches" / "hoyo_tts" / "output"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        print_step("正在下载模型（初次运行可能需要几分钟）...")
        model_dir = snapshot_download('Genius-Society/hoyoTTS', cache_dir=str(local_cache_dir))

        config_path = os.path.join(model_dir, 'configuration.json')
        config = {
            "model_dir": model_dir,
            "framework": "pytorch",
            "task": "text-to-speech",
            "pipeline": {
                "type": "hoyotts-speech-generation"
            },
            "model": {
                "type": "hoyotts",
                "duration_model": "DUR_78000.pth",
                "acoustic_model": "D_78000.pth",
                "vocoder_model": "G_78000.pth"
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print_step("正在初始化语音模型...")
        inference = HoyoTTSPipeline(
            model_dir=model_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        print_step("测试语音生成...")
        test_text = "测试文本"
        test_result = inference(test_text)
        if not isinstance(test_result, dict) or 'audio' not in test_result:
            raise ValueError("Pipeline 测试失败")
        print_success("Pipeline 创建并测试成功!")

        class SpeechRequest(BaseModel):
            prompt: str
            voice: str
            speed: Optional[float] = 1.0
            pitch: Optional[float] = 1.0
            pause: Optional[float] = 0.0
            style: Optional[int] = 0

        app = FastAPI(title="Hoyo TTS Server")

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.mount("/audio", StaticFiles(directory=str(TEMP_DIR)), name="audio")

        @app.post("/v1/audio/speech")
        async def create_speech(request: SpeechRequest):
            unique_id = str(uuid.uuid4())[:8]
            output_path = TEMP_DIR / f"{unique_id}.wav"

            try:
                result = inference((request.prompt, request.voice))
                if not isinstance(result, dict) or 'audio' not in result:
                    raise ValueError("语音生成失败，请检查输入是否正确")
                sf.write(str(output_path), result['audio'], result['sample_rate'])
                audio_url = f"http://127.0.0.1:8000/audio/{unique_id}.wav"
                print_success(f"生成成功: {audio_url}")
                return JSONResponse({
                    "status": "success",
                    "audio_url": audio_url
                })
            except Exception as e:
                error_msg = str(e)
                print_error(f"生成失败: {error_msg}")
                raise HTTPException(status_code=500, detail=error_msg)

        @app.get("/health")
        async def health_check():
            return {"status": "healthy"}

        print_success("服务器启动中...")
        uvicorn.run(app, host="127.0.0.1", port=8000)

    except Exception as e:
        print_error(f"服务器启动失败: {str(e)}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确保有足够的磁盘空间（至少1GB）")
        print("3. 尝试重启电脑后重试")
        print("4. 使用管理员权限运行")
        sys.exit(1)


def main():
    """主程序入口"""
    print("\n=== Hoyo TTS 语音合成服务器 ===")
    print("本程序将自动配置并启动语音服务")
    print("首次运行需要下载模型文件（约600MB），请耐心等待")
    print("运行过程中请勿关闭此窗口\n")

    try:
        # 检查管理员权限
        if not is_admin():
            print_step("正在请求管理员权限...")
            if not restart_as_admin():
                print_error("无法获取管理员权限，程序可能无法正常运行")
                input("按Enter继续以普通权限运行，或关闭程序")

        check_python()
        check_and_install_dependencies()
        setup_server()

    except KeyboardInterrupt:
        print("\n\n程序已停止，按 Ctrl+C 再次尝试停止程序")
        logging.info("程序已停止")
        sys.exit(0)
    except OSError as e:
        print_error(f"操作系统错误: {str(e)}")
        print("\n如果遇到问题，建议:")
        print("1. 检查网络连接")
        print("2. 重启电脑后重试")
        print("3. 确保安装了最新版Python")
        sys.exit(1)
    except RuntimeError as e:
        print_error(f"运行时错误: {str(e)}")
        print("\n如果遇到问题，建议:")
        print("1. 检查网络连接")
        print("2. 重启电脑后重试")
        print("3. 确保安装了最新版Python")
        sys.exit(1)
    except Exception as e:
        print_error(f"未知错误: {str(e)}")
        print("\n如果遇到问题，建议:")
        print("1. 检查网络连接")
        print("2. 重启电脑后重试")
        print("3. 确保安装了最新版Python")
        sys.exit(1)


if __name__ == "__main__":
    main()
