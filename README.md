# LingAnnotation
2025《中文信息处理专题》课程作业：语言信息标注管理系统添加了测评模块的改版。

## 运行帮助
若使用 `uv`，请在将项目克隆到本地后，执行以下命令在本地 `8000` 端口上运行服务器：
```bash
uv run manage.py makemigrations listing evaluation
uv run manage.py migrate
uv run manage.py runserver
```

可以选择在执行以上命令前先执行一次 `uv sync` 来单独观察 `uv` 安装依赖项的速度之快，但该步骤不是必须的。

若不使用 `uv`，请使用 `pip` 安装依赖后运行服务器：
```bash
pip install -r requirements.txt
python manage.py makemigrations listing evaluation
python manage.py migrate
python manage.py runserver
```

注意 `makemigrations` 命令若不指定模块则很可能不会迁移数据库，导致服务器启动后无法进入主页。

## 其他说明

本程序更详细的功能说明提供于相应的实验报告中。

## 环境
Python版本：[3.7.8](https://www.python.org/downloads/release/python-378/)

Django版本：[3.0.5](https://docs.djangoproject.com/en/3.0/)

## 鸣谢
本程序的初版来自[@Dylandjk](https://github.com/Dylandjk)的[myproject](https://github.com/Dylandjk/myproject)项目，为便于管理而设置本仓库，在此表示感谢。

本程序只用于学术研究目的，遵循GPL3.0开源协议。
