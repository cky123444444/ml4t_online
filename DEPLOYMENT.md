# ML Server Demo - Docker Compose 部署指南

这个项目已经被容器化，可以通过 Docker Compose 轻松部署到任何支持 Docker 的云服务商。

## 🏗️ 架构概述

该系统包含两个主要服务：

1. **TorchServe**: 负责机器学习模型推理的服务 (端口: 8080, 8081, 8082)
2. **Main Server**: 主要的 API 服务器，处理数据特征生成并调用 TorchServe (端口: 8000)

## 📋 前置要求

- Docker Engine 20.10+
- Docker Compose v2.0+
- 至少 4GB 可用内存
- 至少 10GB 可用磁盘空间

## 🚀 快速启动

### 1. 生产环境部署

```bash
./deploy.sh
```

### 2. 开发环境部署（推荐）

```bash
# 基础开发模式 - 带文件监控
./deploy-dev.sh

# 或者高级开发模式 - Flask 开发服务器 + 实时重载
./deploy-dev-advanced.sh
```

### 3. 手动部署

生产环境：
```bash
# 构建镜像
docker-compose build

# 启动服务
docker-compose up -d

# 查看状态
docker-compose ps
```

开发环境：
```bash
# 使用文件监控启动
docker-compose up --watch -d

# 或使用开发配置
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --watch -d
```

### 2. 测试部署

```bash
./test_deployment.sh
```

### 3. 停止服务

```bash
./stop.sh
```

或者：

```bash
docker-compose down
```

## 🔧 开发模式说明

项目支持多种开发模式，提供不同级别的开发体验：

### 基础开发模式 (`./deploy-dev.sh`)

- ✅ 文件监控：`src/` 目录变化时自动同步到容器
- ✅ 自动重建：`requirements.txt` 或 `Dockerfile` 变化时重建
- ✅ 模型文件监控：TorchServe 模型文件变化时重建
- ✅ Flask 调试模式启用

### 高级开发模式 (`./deploy-dev-advanced.sh`)

- ✅ 所有基础开发模式功能
- ✅ Flask 开发服务器：带有自动重载
- ✅ 源码挂载：本地文件直接映射到容器
- ✅ 增强调试：详细错误信息和堆栈跟踪

### 手动开发命令

```bash
# 使用 watch 模式启动
docker-compose up --watch

# 使用开发配置文件
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --watch

# 查看特定服务日志
docker-compose logs -f server

# 重新构建单个服务
docker-compose build server
```

## 🔧 配置说明

### 环境变量

主服务支持以下环境变量配置：

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `TS_BASE_URL` | `http://torchserve:8080` | TorchServe 服务地址 |
| `TS_MODEL` | `dragonnet` | 默认模型名称 |
| `HDF_PATH` | `/app/torchserve/data/last_max_window_10000_ochlv.hdf` | 数据文件路径 |
| `HDF_KEY` | `my_data` | HDF 文件中的数据键 |
| `NUM_FEATURES` | `120` | 特征数量 |
| `HISTORY_WINDOW` | `240` | 历史窗口大小 |
| `TS_TIMEOUT` | `30` | TorchServe 请求超时时间 |

### 端口映射

| 服务 | 容器端口 | 主机端口 | 用途 |
|------|----------|----------|------|
| Main Server | 8000 | 8000 | 主要 API |
| TorchServe | 8080 | 8080 | 推理 API |
| TorchServe | 8081 | 8081 | 管理 API |
| TorchServe | 8082 | 8082 | 指标 API |

## 🌐 API 端点

### 主服务 (端口 8000)

- `GET /health` - 健康检查
- `GET /config` - 配置信息
- `POST /predict` - 模型推理

#### 示例请求

```bash
# 健康检查
curl http://localhost:8000/health

# 模型推理
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "dragonnet"}'
```

### TorchServe (端口 8080)

- `GET /ping` - 健康检查
- `GET /models` - 模型列表
- `POST /predictions/{model_name}` - 模型推理

## 📊 监控和日志

### 查看日志

```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f server
docker-compose logs -f torchserve
```

### 健康检查

两个服务都配置了健康检查：

```bash
# 检查服务状态
docker-compose ps

# 查看健康状态
docker inspect ml-server --format='{{.State.Health.Status}}'
docker inspect ml-torchserve --format='{{.State.Health.Status}}'
```

## ☁️ 云服务商部署

### AWS ECS

1. 构建并推送镜像到 ECR
2. 创建 ECS 任务定义
3. 配置服务和负载均衡器

### Google Cloud Run

1. 构建并推送镜像到 Container Registry
2. 部署到 Cloud Run
3. 配置服务间通信

### Azure Container Instances

1. 构建并推送镜像到 Container Registry
2. 使用 ACI 部署容器组
3. 配置网络和存储

### 数字海洋 (DigitalOcean)

1. 使用 App Platform 或 Kubernetes
2. 配置 Container Registry
3. 设置负载均衡和域名

## 🔒 生产环境注意事项

### 安全性

1. **移除调试模式**: 在生产环境中设置 `debug=False`
2. **使用 HTTPS**: 配置 SSL/TLS 证书
3. **限制端口访问**: 只暴露必要的端口
4. **环境变量**: 使用密钥管理系统存储敏感信息

### 性能优化

1. **资源限制**: 设置适当的 CPU 和内存限制
2. **健康检查间隔**: 根据需要调整健康检查频率
3. **日志轮转**: 配置日志轮转以防止磁盘空间耗尽
4. **缓存**: 考虑添加 Redis 等缓存层

### 高可用性

1. **多实例**: 运行多个服务实例
2. **负载均衡**: 使用负载均衡器分发流量
3. **自动重启**: 配置自动重启策略
4. **监控**: 设置监控和报警

## 🐛 故障排除

### 常见问题

1. **端口冲突**: 确保端口 8000, 8080-8082 没有被占用
2. **内存不足**: 确保有足够的可用内存 (至少 4GB)
3. **TorchServe 启动慢**: 首次启动可能需要几分钟加载模型
4. **网络连接**: 确保容器间网络连接正常

### 调试命令

```bash
# 查看容器状态
docker-compose ps

# 进入容器进行调试
docker-compose exec server bash
docker-compose exec torchserve bash

# 查看详细日志
docker-compose logs --tail=100 -f server

# 重新构建并启动
docker-compose up --build -d
```

## 📚 更多资源

- [Docker Compose 文档](https://docs.docker.com/compose/)
- [TorchServe 文档](https://pytorch.org/serve/)
- [Flask 部署指南](https://flask.palletsprojects.com/en/2.3.x/deploying/)

## 🆘 获取帮助

如果遇到问题，请检查：

1. 日志文件中的错误信息
2. 容器健康状态
3. 网络连接
4. 资源使用情况

执行 `./test_deployment.sh` 可以进行完整的功能测试。